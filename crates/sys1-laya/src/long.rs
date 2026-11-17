//! Long-context scanning: split a tokenized state into overlapping windows,
//! score each window, and aggregate the per-question answers.

use crate::config::LayaConfig;
use crate::prepare::{
    PreparedRequest, prepare_request_with_state_ids, state_room, state_token_ids,
};
use crate::tokenizer::LayaTokenizer;
use indexmap::IndexMap;
use sys1_core::{
    Answers, DecisionOutput, LongContextOptions, ModelError, Questions, WindowAttribution,
};
use sys1_protocol::{Answer, Question, State, Usage};

/// The window layout for one long-context call.
pub(crate) struct LongPlan {
    pub state_ids: Vec<u32>,
    pub windows: Vec<(usize, usize)>,
    pub window: usize,
    pub stride: usize,
    pub single: bool,
}

/// Size the windows against the room the questions actually leave.
///
/// The default window is `max(64, max_len - head_max_len - 8)`. An explicit
/// window is capped at the smallest state room across the questions, because
/// one list of windows is scored for every question. A state that fits the
/// room (or an explicit window) takes the normal single-call path.
pub(crate) fn plan(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    state: &State,
    questions: &Questions,
    options: LongContextOptions,
) -> Result<LongPlan, ModelError> {
    if questions.is_empty() {
        return Err(ModelError::invalid_request("questions must not be empty"));
    }

    let state_ids = state_token_ids(tokenizer, state)?;
    let mut room = usize::MAX;
    for (id, question) in questions.iter() {
        room = room.min(state_room(tokenizer, config, id, question)?);
    }
    if room == 0 {
        return Err(ModelError::invalid_request(format!(
            "predict_long: the questions' options fill max_len={} (head_max_len={}), \
             leaving no room for the state",
            config.max_len, config.head_max_len
        )));
    }

    let explicit_window = options.window.filter(|window| *window > 0);
    let requested = explicit_window.unwrap_or_else(|| {
        config
            .max_len
            .saturating_sub(config.head_max_len + 8)
            .max(64)
    });
    let window = requested.min(room).max(1);
    if explicit_window.is_some_and(|requested| requested > window) {
        tracing::warn!(
            requested,
            window,
            room,
            "predict_long window is wider than the questions' state room; scanning with the smaller window"
        );
    }

    let stride = match options.stride.filter(|stride| *stride > 0) {
        Some(stride) => {
            if stride > requested {
                return Err(ModelError::invalid_request(format!(
                    "predict_long: stride={stride} exceeds the requested window={requested}; \
                     tokens between windows would be skipped"
                )));
            }
            if stride > window {
                (window / 2).max(1)
            } else {
                stride
            }
        }
        None => (window / 2).max(1),
    };

    let single_limit = explicit_window.unwrap_or(room);
    let single = state_ids.len() <= single_limit;
    let windows = if single {
        Vec::new()
    } else {
        split(state_ids.len(), window, stride)
    };

    Ok(LongPlan {
        state_ids,
        windows,
        window,
        stride,
        single,
    })
}

fn split(length: usize, window: usize, stride: usize) -> Vec<(usize, usize)> {
    let mut windows = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + window).min(length);
        windows.push((start, end));
        if end == length {
            break;
        }
        start += stride;
    }
    windows
}

/// Prepare one request per window, each carrying the same question set.
pub(crate) fn prepare_windows(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    plan: &LongPlan,
    questions: &Questions,
) -> Result<Vec<PreparedRequest>, ModelError> {
    let mut prepared = Vec::with_capacity(plan.windows.len());
    for &(start, end) in &plan.windows {
        prepared.push(prepare_request_with_state_ids(
            tokenizer,
            config,
            &plan.state_ids[start..end],
            questions,
        )?);
    }
    Ok(prepared)
}

/// Score prepared windows in chunks bounded by `batch_size`.
pub(crate) fn score_windows(
    prepared: Vec<PreparedRequest>,
    batch_size: Option<usize>,
    mut score: impl FnMut(Vec<PreparedRequest>) -> Vec<Result<DecisionOutput, ModelError>>,
) -> Result<Vec<DecisionOutput>, ModelError> {
    let chunk = batch_size
        .filter(|size| *size > 0)
        .unwrap_or(prepared.len());
    let mut prepared = prepared;
    let mut outputs = Vec::with_capacity(prepared.len());
    while !prepared.is_empty() {
        let take = chunk.min(prepared.len()).max(1);
        let batch: Vec<PreparedRequest> = prepared.drain(..take).collect();
        for output in score(batch) {
            outputs.push(output?);
        }
    }
    Ok(outputs)
}

/// Aggregate per-window outputs.
///
/// `noul` takes the window with the largest P(true). `choice` and `score`
/// take the single most-confident window, measured by the largest answer
/// probability, so a localized signal is not out-voted by neutral windows.
/// The deciding window's action probability and full answer are carried
/// through; `usage.input_tokens` sums every scored window.
pub(crate) fn aggregate(
    outputs: Vec<DecisionOutput>,
    questions: &Questions,
    windows: &[(usize, usize)],
) -> Result<DecisionOutput, ModelError> {
    let count = outputs.len();
    let mut answers = IndexMap::with_capacity(questions.len());
    let mut actions = IndexMap::new();
    let mut attribution = IndexMap::new();

    for (id, question) in questions.iter() {
        let mut best: Option<(usize, f64)> = None;
        for (index, output) in outputs.iter().enumerate() {
            let answer = output.answers.get(id).ok_or_else(|| {
                ModelError::inference(format!(
                    "predict_long: window {index} produced no answer for question {id:?}"
                ))
            })?;
            let score = deciding_score(question, answer)?;
            if best.is_none_or(|(_, current)| score > current) {
                best = Some((index, score));
            }
        }
        let (index, _) = best.expect("questions are non-empty and every window answered");
        let answer = outputs[index]
            .answers
            .get(id)
            .expect("deciding window was checked above")
            .clone();
        answers.insert(id.clone(), answer);
        if let Some(action) = outputs[index].action(id) {
            actions.insert(id.clone(), action);
        }
        if let Some(&(start, end)) = windows.get(index) {
            attribution.insert(
                id.clone(),
                WindowAttribution {
                    index,
                    token_start: start,
                    token_end: end,
                    count,
                },
            );
        }
    }

    let input_tokens = outputs.iter().map(|output| output.usage.input_tokens).sum();
    let output_tokens = outputs
        .iter()
        .map(|output| output.usage.output_tokens)
        .sum();
    Ok(DecisionOutput::new(
        Answers::new(answers),
        Usage::new(input_tokens, output_tokens),
    )
    .with_actions(actions)
    .with_windows(attribution, count))
}

fn deciding_score(question: &Question, answer: &Answer) -> Result<f64, ModelError> {
    match (question, answer) {
        (Question::Noul(_), Answer::Noul(answer)) => Ok(answer.noul),
        (Question::Choice(_), Answer::Choice(answer)) => {
            Ok(max_probability(&answer.probabilities, answer.confidence))
        }
        (Question::Score(_), Answer::Score(answer)) => {
            Ok(max_probability(&answer.probabilities, answer.confidence))
        }
        (question, answer) => Err(ModelError::inference(format!(
            "predict_long: question type {:?} produced answer type {:?}",
            question.question_type(),
            answer.answer_type()
        ))),
    }
}

fn max_probability(probabilities: &IndexMap<String, f64>, confidence: f64) -> f64 {
    let max = probabilities
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max.is_finite() { max } else { confidence }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sys1_core::{Answers, Questions};
    use sys1_protocol::{Answer, ChoiceAnswer, NoulAnswer, Usage};

    fn choice_probabilities(values: &[(&str, f64)]) -> IndexMap<String, f64> {
        values
            .iter()
            .map(|(label, probability)| ((*label).to_owned(), *probability))
            .collect()
    }

    fn window_output(
        billing: f64,
        tone: IndexMap<String, f64>,
        action: f32,
        input_tokens: u64,
    ) -> DecisionOutput {
        let answers = IndexMap::from([
            (
                "billing".to_owned(),
                Answer::Noul(NoulAnswer { noul: billing }),
            ),
            (
                "tone".to_owned(),
                Answer::Choice(ChoiceAnswer {
                    choice: "calm".to_owned(),
                    confidence: tone.values().copied().fold(0.0, f64::max),
                    probabilities: tone,
                }),
            ),
        ]);
        DecisionOutput::new(Answers::new(answers), Usage::new(input_tokens, 0))
            .with_actions(IndexMap::from([("tone".to_owned(), action)]))
    }

    #[test]
    fn aggregates_noul_by_max_and_choice_by_most_confident_window() {
        let questions = Questions::new()
            .noul("billing", "Is this about billing?")
            .choice("tone", [("angry", "Upset"), ("calm", "Neutral")]);
        let outputs = vec![
            window_output(
                0.2,
                choice_probabilities(&[("angry", 0.1), ("calm", 0.9)]),
                0.1,
                10,
            ),
            window_output(
                0.8,
                choice_probabilities(&[("angry", 0.6), ("calm", 0.4)]),
                0.9,
                20,
            ),
        ];
        let aggregated = aggregate(outputs, &questions, &[(0, 10), (4, 14)]).expect("aggregate");

        assert_eq!(aggregated.answers.noul("billing").expect("noul").noul, 0.8);
        assert_eq!(
            aggregated.answers.choice("tone").expect("choice").choice,
            "calm"
        );
        assert_eq!(aggregated.action("tone"), Some(0.1));
        assert_eq!(aggregated.scored_windows, 2);
        assert_eq!(aggregated.usage.input_tokens, 30);
        assert_eq!(
            aggregated.window("billing").expect("billing window").index,
            1
        );
        assert_eq!(aggregated.window("tone").expect("tone window").index, 0);
        assert_eq!(
            aggregated.window("tone").expect("tone window").token_start,
            0
        );
        assert_eq!(aggregated.window("tone").expect("tone window").count, 2);
    }

    #[test]
    fn splits_with_overlap_until_the_tail_is_covered() {
        assert_eq!(split(10, 4, 2), vec![(0, 4), (2, 6), (4, 8), (6, 10)]);
        assert_eq!(split(4, 4, 2), vec![(0, 4)]);
        assert_eq!(split(5, 4, 4), vec![(0, 4), (4, 5)]);
    }

    #[test]
    fn every_token_is_covered_when_stride_does_not_exceed_window() {
        let windows = split(37, 8, 3);
        let mut covered = vec![false; 37];
        for (start, end) in windows {
            for slot in covered.iter_mut().take(end).skip(start) {
                *slot = true;
            }
        }
        assert!(covered.into_iter().all(|value| value));
    }
}
