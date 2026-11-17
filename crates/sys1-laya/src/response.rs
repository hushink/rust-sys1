use crate::calibration::{ConfidenceMode, argmax, confidence, probability, round4};
use crate::config::LayaConfig;
use crate::prepare::{Criteria, Item, PreparedRequest};
use indexmap::IndexMap;
use sys1_core::{Answers, DecisionOutput};
use sys1_protocol::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, Usage};

/// Format one prepared request from one logit vector per item.
pub(crate) fn build_response(
    config: &LayaConfig,
    confidence_mode: ConfidenceMode,
    request: PreparedRequest,
    logits: &mut impl Iterator<Item = (Vec<f32>, Option<f32>)>,
) -> DecisionOutput {
    let input_tokens = request.items.iter().map(|item| item.ids.len() as u64).sum();
    let mut answers: IndexMap<String, Answer> = IndexMap::with_capacity(request.items.len());
    let mut actions = IndexMap::new();

    for item in request.items {
        let (values, action) = logits.next().expect("one logits vector per item");
        let temperature = config.temperature_for(item.question.kind, values.len());
        let probabilities = probability(&values, temperature);
        let conf = confidence(confidence_mode, &probabilities);

        if let Some(action) = action {
            actions.insert(item.question.id.clone(), round4(action as f64) as f32);
        }
        let answer = answer_for(&item, &probabilities, conf);
        answers.insert(item.question.id, answer);
    }

    DecisionOutput::new(Answers::new(answers), Usage::new(input_tokens, 0)).with_actions(actions)
}

fn answer_for(item: &Item, probabilities: &[f32], conf: f64) -> Answer {
    match item.question.kind {
        0 => {
            let index = argmax(probabilities);
            let mut distribution = IndexMap::with_capacity(probabilities.len());
            for (label, value) in item.question.labels.iter().zip(probabilities.iter()) {
                distribution.insert(label.clone(), round4(*value as f64));
            }
            Answer::Choice(ChoiceAnswer::new(
                item.question.labels[index].clone(),
                round4(conf),
                distribution,
            ))
        }
        1 => {
            let score: f32 = probabilities
                .iter()
                .enumerate()
                .map(|(index, value)| index as f32 * value)
                .sum();
            let mut distribution = IndexMap::with_capacity(probabilities.len());
            let mut legend = IndexMap::with_capacity(probabilities.len());
            for (index, value) in probabilities.iter().enumerate() {
                distribution.insert(index.to_string(), round4(*value as f64));
                if let Criteria::Score(criteria) = &item.question.criteria
                    && let Some(content) = criteria.get(index)
                {
                    legend.insert(index.to_string(), content.clone());
                }
            }
            Answer::Score(ScoreAnswer::new(
                round4(score as f64),
                round4(conf),
                legend,
                distribution,
            ))
        }
        _ => {
            let true_probability = probabilities.get(1).copied().unwrap_or(0.0);
            Answer::Noul(NoulAnswer::new(round4(true_probability as f64)))
        }
    }
}
