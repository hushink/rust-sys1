use indexmap::IndexMap;
use std::path::PathBuf;
use sys1_core::{DecisionModel, Questions};
use sys1_decima::{DecimaModel, DecimaOptions};
use sys1_protocol::{ChoiceQuestion, Content, Question, State};

const LONG_INSTRUCTION: &str = "A customer writes at length about a duplicate annual subscription charge, an \
unconfirmed renewal, two support tickets, a promised refund that never arrived, a failed card update, and a \
request to escalate the case to a supervisor before the next billing cycle begins. Which part of the account \
is this about?";

#[test]
#[ignore = "requires SYS1_DECIMA_TEST_DIR pointing at a decima-base repository"]
fn predicts_all_question_types() {
    let directory =
        PathBuf::from(std::env::var("SYS1_DECIMA_TEST_DIR").expect("SYS1_DECIMA_TEST_DIR"));
    let model = DecimaModel::load_dir(&directory, DecimaOptions::default()).unwrap();
    let questions = Questions::new()
        .choice("team", [("billing", "charges, refunds"), ("tech", "bugs")])
        .noul("urgent", "The customer needs an answer today.")
        .score("anger", ["calm", "annoyed", "furious"]);
    let output = model
        .predict(&State::text("My card was charged twice"), &questions)
        .unwrap();
    let team = output.answers.choice("team").unwrap();
    assert_eq!(team.choice, "billing");
    for (label, want) in [("billing", 0.972121), ("tech", 0.027879)] {
        let got = team.probabilities[label];
        assert!((got - want).abs() < 1e-3, "{label}: {got} vs {want}");
    }
    let urgent = output.answers.noul("urgent").unwrap();
    assert!((urgent.noul - 0.644244).abs() < 1e-3, "{}", urgent.noul);
    let anger = output.answers.score("anger").unwrap();
    assert!((anger.score - 0.946362).abs() < 1e-3, "{}", anger.score);
}

#[test]
#[ignore = "requires SYS1_DECIMA_SMALL_TEST_DIR pointing at decima-small"]
fn predicts_with_the_bert_encoder() {
    let directory = PathBuf::from(
        std::env::var("SYS1_DECIMA_SMALL_TEST_DIR").expect("SYS1_DECIMA_SMALL_TEST_DIR"),
    );
    let model = DecimaModel::load_dir(&directory, DecimaOptions::default()).unwrap();
    let questions = Questions::new()
        .choice("team", [("billing", "charges, refunds"), ("tech", "bugs")])
        .noul("urgent", "The customer needs an answer today.")
        .score("anger", ["calm", "annoyed", "furious"]);
    let output = model
        .predict(&State::text("My card was charged twice"), &questions)
        .unwrap();
    let team = output.answers.choice("team").unwrap();
    assert_eq!(team.choice, "billing");
    for (label, want) in [("billing", 0.984525), ("tech", 0.015475)] {
        let got = team.probabilities[label];
        assert!((got - want).abs() < 1e-3, "{label}: {got} vs {want}");
    }
    let urgent = output.answers.noul("urgent").unwrap();
    assert!((urgent.noul - 0.235711).abs() < 1e-3, "{}", urgent.noul);
    let anger = output.answers.score("anger").unwrap();
    assert!((anger.score - 0.798843).abs() < 1e-3, "{}", anger.score);
    for (index, want) in [(0, 0.309643), (1, 0.581871), (2, 0.108486)] {
        let got = anger.probabilities[&index.to_string()];
        assert!((got - want).abs() < 1e-3, "{index}: {got} vs {want}");
    }
}

#[test]
#[ignore = "requires SYS1_DECIMA_GUARD_SMALL_TEST_DIR pointing at decima-guard-small"]
fn predicts_with_the_guard_small_checkpoint() {
    let directory = PathBuf::from(
        std::env::var("SYS1_DECIMA_GUARD_SMALL_TEST_DIR")
            .expect("SYS1_DECIMA_GUARD_SMALL_TEST_DIR"),
    );
    let model = DecimaModel::load_dir(&directory, DecimaOptions::default()).unwrap();
    let state = State::text("My card was charged twice");

    let questions = Questions::new()
        .choice("team", [("billing", "charges, refunds"), ("tech", "bugs")])
        .noul("urgent", "The customer needs an answer today.")
        .score("anger", ["calm", "annoyed", "furious"]);
    let output = model.predict(&state, &questions).unwrap();
    let team = output.answers.choice("team").unwrap();
    assert_eq!(team.choice, "billing");
    for (label, want) in [("billing", 0.996044), ("tech", 0.003956)] {
        let got = team.probabilities[label];
        assert!((got - want).abs() < 1e-4, "{label}: {got} vs {want}");
    }
    let urgent = output.answers.noul("urgent").unwrap();
    assert!((urgent.noul - 0.414360).abs() < 1e-4, "{}", urgent.noul);
    let anger = output.answers.score("anger").unwrap();
    assert!((anger.score - 0.783849).abs() < 1e-4, "{}", anger.score);
    for (index, want) in [(0, 0.429440), (1, 0.357271), (2, 0.213289)] {
        let got = anger.probabilities[&index.to_string()];
        assert!((got - want).abs() < 1e-4, "{index}: {got} vs {want}");
    }

    // A question that does not fit the 64-token option window: Decima v2's
    // `choice_fit` keeps each option whole and shortens the question beside it.
    let criteria: IndexMap<String, Content> = [
        ("billing", "the duplicate charge and the refund"),
        (
            "technical",
            "the card update failure and the broken account page",
        ),
    ]
    .into_iter()
    .map(|(label, description)| (label.to_owned(), Content::String(description.to_owned())))
    .collect();
    let questions = Questions::new().insert(
        "topic",
        Question::Choice(ChoiceQuestion::new(criteria).with_instructions(LONG_INSTRUCTION)),
    );
    let output = model.predict(&state, &questions).unwrap();
    let topic = output.answers.choice("topic").unwrap();
    assert_eq!(topic.choice, "billing");
    for (label, want) in [("billing", 0.920987), ("technical", 0.079013)] {
        let got = topic.probabilities[label];
        assert!((got - want).abs() < 1e-4, "{label}: {got} vs {want}");
    }
}
