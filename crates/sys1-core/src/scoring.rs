//! Shared probability and answer construction for model families.

use indexmap::IndexMap;
use sys1_protocol::{Answer, ChoiceAnswer, Content, NoulAnswer, Question, ScoreAnswer};

/// Softmax with a positive temperature.
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temperature = temperature.max(f32::MIN_POSITIVE);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    let mut values: Vec<f32> = logits
        .iter()
        .map(|value| (value / temperature - max).exp())
        .collect();
    let total: f32 = values.iter().sum();
    if total > 0.0 {
        values.iter_mut().for_each(|value| *value /= total);
    }
    values
}

/// Index of the first maximum, as Python's `argmax`.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// TypeSafe's normalized top probability `(K * max(p) - 1) / (K - 1)`.
pub fn typesafe_confidence(probabilities: &IndexMap<String, f64>) -> f64 {
    let count = probabilities.len();
    if count < 2 {
        return 1.0;
    }
    let max = probabilities
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    ((count as f64 * max - 1.0) / (count as f64 - 1.0)).clamp(0.0, 1.0)
}

pub fn choice_answer(labels: &[String], probabilities: &[f32]) -> Answer {
    let best = argmax(probabilities);
    let probabilities = labels
        .iter()
        .zip(probabilities)
        .map(|(label, probability)| (label.clone(), f64::from(*probability)))
        .collect::<IndexMap<_, _>>();
    Answer::Choice(ChoiceAnswer {
        choice: labels[best].clone(),
        confidence: typesafe_confidence(&probabilities),
        probabilities,
    })
}

pub fn score_answer(legend: &IndexMap<String, Content>, probabilities: &[f32]) -> Answer {
    let probabilities = (0..legend.len().min(probabilities.len()))
        .map(|index| (index.to_string(), f64::from(probabilities[index])))
        .collect::<IndexMap<_, _>>();
    let score = probabilities
        .values()
        .enumerate()
        .map(|(index, probability)| index as f64 * probability)
        .sum();
    Answer::Score(ScoreAnswer {
        score,
        confidence: typesafe_confidence(&probabilities),
        legend: legend.clone(),
        probabilities,
    })
}

pub fn noul_answer(_false_probability: f32, true_probability: f32) -> Answer {
    Answer::Noul(NoulAnswer {
        noul: f64::from(true_probability).clamp(0.0, 1.0),
    })
}

/// Build an answer from logits in answer order: choice labels, score levels, or
/// noul `[false, true]`.
pub fn answer_from_logits(
    question: &Question,
    labels: &[String],
    legend: &IndexMap<String, Content>,
    logits: &[f32],
    temperature: f32,
) -> Answer {
    let probabilities = softmax(logits, temperature);
    match question {
        Question::Choice(_) => choice_answer(labels, &probabilities),
        Question::Score(_) => score_answer(legend, &probabilities),
        Question::Noul(_) => noul_answer(
            probabilities.first().copied().unwrap_or(0.5),
            probabilities.get(1).copied().unwrap_or(0.5),
        ),
        Question::Box(_) => unreachable!("box questions are handled by the caller"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn probabilities_and_confidence() {
        let probabilities = softmax(&[0.0, 0.0, 0.0], 1.0);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let map = IndexMap::from([
            ("a".to_owned(), 0.6),
            ("b".to_owned(), 0.3),
            ("c".to_owned(), 0.1),
        ]);
        assert!((typesafe_confidence(&map) - 0.4).abs() < 1e-12);
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    }

    #[test]
    fn answers_use_answer_order() {
        let labels = ["a".to_owned(), "b".to_owned()];
        let answer = choice_answer(&labels, &[0.25, 0.75]);
        let Answer::Choice(answer) = answer else {
            panic!("choice");
        };
        assert_eq!(answer.choice, "b");
        let legend = IndexMap::from([
            ("0".to_owned(), "low".into()),
            ("1".to_owned(), "high".into()),
        ]);
        let Answer::Score(answer) = score_answer(&legend, &[0.25, 0.75]) else {
            panic!("score");
        };
        assert!((answer.score - 0.75).abs() < 1e-12);
        let Answer::Noul(answer) = noul_answer(0.2, 0.8) else {
            panic!("noul");
        };
        assert!((answer.noul - 0.8).abs() < 1e-6);
        let _ = json!({});
    }
}
