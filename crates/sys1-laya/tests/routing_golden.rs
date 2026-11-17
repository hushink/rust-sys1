//! Replay the official Laya routing golden vectors.
//!
//! Set `SYS1_LAYA_ROUTING_GOLDEN` to the official
//! `laya-rust/tests/golden/routing/route_probe.json` file. Without it the
//! test prints a skip line.

use indexmap::IndexMap;
use serde_json::Value;
use std::fs;
use std::path::Path;
use sys1_core::Questions;
use sys1_laya::{LayaOptions, LayaRouter, LayaRouterOptions, RouteHints, RouteVariant};
use sys1_protocol::{Question, State};

fn state_from(value: &Value) -> State {
    match value {
        Value::String(_) | Value::Object(_) | Value::Array(_) => {
            State::from_value(value.clone()).expect("valid state")
        }
        _ => State::text(""),
    }
}

fn questions_from(value: Option<&Value>) -> Questions {
    match value {
        Some(Value::Object(map)) => {
            let questions: IndexMap<String, Question> =
                serde_json::from_value(Value::Object(map.clone())).expect("questions");
            Questions::from(questions)
        }
        _ => Questions::from(IndexMap::new()),
    }
}

fn router(options: &Value) -> LayaRouter {
    let mut router = LayaRouter::with_paths(
        "/nonexistent/english",
        "/nonexistent/multilingual",
        "/nonexistent/typed-decisions",
        LayaOptions::default(),
    );
    let default = options
        .get("default")
        .and_then(Value::as_str)
        .map(|value| RouteVariant::from_name(value).expect("default variant"))
        .unwrap_or(RouteVariant::Multilingual);
    let auto_task_detection = options
        .get("auto_task_detection")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    router.set_routing(LayaRouterOptions {
        default,
        auto_task_detection,
    });
    if let Some(guess) = options.get("lang_guess").and_then(Value::as_str) {
        router.set_lang_guess(Some(guess.to_owned()));
    }
    router
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-4
}

fn compare_detection(label: &str, expected: &Value, actual: Option<&sys1_laya::LanguageAnalysis>) {
    let expected = match expected {
        Value::Null => {
            assert!(actual.is_none(), "{label}: expected no detection");
            return;
        }
        Value::Object(value) => value,
        _ => panic!("{label}: invalid expected detection"),
    };
    let actual = actual.unwrap_or_else(|| panic!("{label}: expected detection"));
    assert_eq!(
        actual.script,
        expected["script"].as_str().expect("script"),
        "{label}.script"
    );
    assert_eq!(
        actual.language.as_deref(),
        expected["language"].as_str(),
        "{label}.language"
    );
    assert_eq!(
        actual.is_english,
        expected["is_english"].as_bool().expect("is_english"),
        "{label}.is_english"
    );
    assert_eq!(
        actual.language_undecided,
        expected["language_undecided"]
            .as_bool()
            .expect("language_undecided"),
        "{label}.language_undecided"
    );
    assert_eq!(
        actual.mixed_segment.as_deref(),
        expected["mixed_segment"].as_str(),
        "{label}.mixed_segment"
    );
    assert!(
        close(
            actual.diacritic_rate,
            expected["diacritic_rate"].as_f64().expect("diacritic_rate")
        ),
        "{label}.diacritic_rate: {} vs {}",
        actual.diacritic_rate,
        expected["diacritic_rate"]
    );
    assert!(
        close(
            actual.non_latin_fraction,
            expected["non_latin_fraction"]
                .as_f64()
                .expect("non_latin_fraction")
        ),
        "{label}.non_latin_fraction: {} vs {}",
        actual.non_latin_fraction,
        expected["non_latin_fraction"]
    );
    for (script, value) in expected["script_profile"]
        .as_object()
        .expect("script_profile")
    {
        let actual = actual
            .script_profile
            .get(script)
            .unwrap_or_else(|| panic!("{label}.script_profile missing {script}"));
        assert!(
            close(*actual, value.as_f64().expect("profile value")),
            "{label}.script_profile.{script}"
        );
    }
}

#[test]
fn matches_the_official_routing_golden() {
    let Some(path) = std::env::var_os("SYS1_LAYA_ROUTING_GOLDEN") else {
        eprintln!("skipping routing golden (set SYS1_LAYA_ROUTING_GOLDEN)");
        return;
    };
    let path = Path::new(&path);
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(path).expect("routing golden JSON"))
        .expect("routing golden array");
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for case in &cases {
        let label = case["label"].as_str().unwrap_or("?");
        let route = &case["route_kwargs"];
        let state = state_from(&route["state"]);
        let questions = questions_from(route.get("questions"));
        let hints = RouteHints {
            model: route
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
            task: route.get("task").and_then(Value::as_str).map(str::to_owned),
            lang: route.get("lang").and_then(Value::as_str).map(str::to_owned),
            lang_guess: route
                .get("lang_guess")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let router = router(&case["router_kwargs"]);
        let actual = match router.route_with(&state, &questions, &hints) {
            Ok(decision) => decision,
            Err(error) => {
                failures.push(format!("{label}: route error: {error}"));
                continue;
            }
        };
        let expected = &case["decision"];
        let expected_model = expected["model"].as_str().expect("model");
        if actual.model.as_str() != expected_model {
            failures.push(format!(
                "{label}: model expected {expected_model}, got {} ({})",
                actual.model.as_str(),
                actual.reason
            ));
        }
        if actual.workflow.as_deref() != expected["workflow"].as_str() {
            failures.push(format!(
                "{label}: workflow expected {:?}, got {:?}",
                expected["workflow"], actual.workflow
            ));
        }
        if actual.reason != expected["reason"].as_str().unwrap_or("") {
            failures.push(format!(
                "{label}: reason expected {:?}, got {:?}",
                expected["reason"], actual.reason
            ));
        }
        compare_detection(label, &expected["detection"], actual.detection.as_ref());
        checked += 1;
    }
    if !failures.is_empty() {
        for failure in failures.iter().take(40) {
            eprintln!("{failure}");
        }
        panic!(
            "{} of {checked} routing cases failed ({} total)",
            failures.len(),
            cases.len()
        );
    }
    eprintln!("routing golden: {checked} cases matched");
}
