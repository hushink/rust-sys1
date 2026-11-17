use serde_json::{Map, Value, json};
use sys1_protocol::*;

fn parse(value: Value) -> Result<SystemOneRequest, ValidationReport> {
    parse_request_value(value)
}

#[test]
fn official_noul_request_roundtrip() {
    let value = json!({
        "model": "jev-latest",
        "state": "I was charged twice for my subscription this month. Please refund the extra charge.",
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this message about billing?"
            }
        }
    });

    let request = parse(value.clone()).expect("valid noul request");
    assert_eq!(request.model, "jev-latest");
    assert_eq!(request.questions.len(), 1);

    let Question::Noul(noul) = &request.questions["billing"] else {
        panic!("expected noul question");
    };
    assert_eq!(
        noul.instructions.as_ref().and_then(Content::as_str),
        Some("Is this message about billing?")
    );

    assert_eq!(serde_json::to_value(&request).unwrap(), value);
}

#[test]
fn official_choice_request_preserves_criteria_order() {
    let value = json!({
        "model": "jev-latest",
        "state": "I was charged twice for my subscription this month. Please refund the extra charge.",
        "questions": {
            "tone": {
                "type": "choice",
                "instructions": "What is the tone of this message?",
                "criteria": {
                    "angry": "An upset or hostile message",
                    "calm": "A neutral or polite message",
                    "excited": "An enthusiastic or eager message"
                }
            }
        }
    });

    let request = parse(value.clone()).expect("valid choice request");
    let Question::Choice(choice) = &request.questions["tone"] else {
        panic!("expected choice question");
    };
    assert_eq!(
        choice.criteria.keys().cloned().collect::<Vec<_>>(),
        vec!["angry", "calm", "excited"]
    );
    assert_eq!(serde_json::to_value(&request).unwrap(), value);
}

#[test]
fn official_score_request_roundtrip() {
    let value = json!({
        "model": "jev-latest",
        "state": "I was charged twice for my subscription this month. Please refund the extra charge.",
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this message?",
                "criteria": [
                    "Can wait",
                    "Needs attention this week",
                    "Needs attention today"
                ]
            }
        }
    });

    let request = parse(value.clone()).expect("valid score request");
    let Question::Score(score) = &request.questions["urgency"] else {
        panic!("expected score question");
    };
    assert_eq!(score.criteria.len(), 3);
    assert_eq!(serde_json::to_value(&request).unwrap(), value);
}

#[test]
fn official_box_request_with_asset_roundtrip() {
    let value = json!({
        "model": "jev-preview",
        "state": "photo",
        "questions": {
            "red_rectangle": {
                "type": "box",
                "instructions": "Locate the red rectangle."
            }
        },
        "assets": {
            "photo": {
                "type": "image",
                "format": "png",
                "data_base64": "iVBORw0KGgo="
            }
        }
    });

    let request = parse(value.clone()).expect("valid box request");
    assert!(request.assets.as_ref().unwrap().contains_key("photo"));
    assert_eq!(serde_json::to_value(&request).unwrap(), value);
}

#[test]
fn official_responses_parse_and_roundtrip() {
    let noul_value = json!({
        "model": "jev-latest",
        "answers": { "billing": { "type": "noul", "noul": 0.98 } },
        "usage": { "input_tokens": 120, "output_tokens": 12 }
    });
    let noul: SystemOneResponse =
        serde_json::from_value(noul_value.clone()).expect("valid noul response");
    assert!(matches!(noul.answers["billing"], Answer::Noul(_)));
    assert_eq!(serde_json::to_value(&noul).unwrap(), noul_value);

    let choice_value = json!({
        "model": "jev-latest",
        "answers": {
            "tone": {
                "type": "choice",
                "choice": "calm",
                "confidence": 0.7,
                "probabilities": { "angry": 0.15, "calm": 0.8, "excited": 0.05 }
            }
        },
        "usage": { "input_tokens": 120, "output_tokens": 12 }
    });
    let choice: SystemOneResponse =
        serde_json::from_value(choice_value.clone()).expect("valid choice response");
    let Answer::Choice(answer) = &choice.answers["tone"] else {
        panic!("expected choice answer");
    };
    assert_eq!(answer.choice, "calm");
    assert_eq!(answer.probabilities["calm"], 0.8);
    assert_eq!(serde_json::to_value(&choice).unwrap(), choice_value);

    let score_value = json!({
        "model": "jev-latest",
        "answers": {
            "urgency": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.55,
                "legend": {
                    "0": "Can wait",
                    "1": "Needs attention this week",
                    "2": "Needs attention today"
                },
                "probabilities": { "0": 0.1, "1": 0.1, "2": 0.8 }
            }
        },
        "usage": { "input_tokens": 120, "output_tokens": 12 }
    });
    let score: SystemOneResponse =
        serde_json::from_value(score_value.clone()).expect("valid score response");
    let Answer::Score(answer) = &score.answers["urgency"] else {
        panic!("expected score answer");
    };
    assert_eq!(answer.score, 1.7);
    assert_eq!(answer.legend.len(), 3);
    assert_eq!(serde_json::to_value(&score).unwrap(), score_value);

    let box_value = json!({
        "model": "jev-preview",
        "answers": {
            "red_rectangle": {
                "type": "box",
                "box": { "top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68 },
                "confidence": null
            }
        },
        "usage": { "input_tokens": 285, "output_tokens": 20 }
    });
    let boxed: SystemOneResponse =
        serde_json::from_value(box_value.clone()).expect("valid box response");
    let Answer::Box(answer) = &boxed.answers["red_rectangle"] else {
        panic!("expected box answer");
    };
    assert_eq!(answer.confidence, None);
    assert_eq!(serde_json::to_value(&boxed).unwrap(), box_value);
}

#[test]
fn models_response_requires_all_fields() {
    let value = json!({
        "models": [{
            "name": "jev-latest",
            "description": "General-purpose system one model.",
            "release_date": "2026-09-15"
        }]
    });
    let list: ModelMetadataList = serde_json::from_value(value.clone()).expect("valid model list");
    assert_eq!(list.models[0].name, "jev-latest");
    assert_eq!(serde_json::to_value(&list).unwrap(), value);

    let missing = serde_json::from_value::<ModelMetadataList>(json!({
        "models": [{ "name": "jev-latest", "description": "x" }]
    }));
    assert!(missing.is_err());
}

#[test]
fn state_rejects_null_number_and_bool() {
    for invalid in [Value::Null, json!(1), json!(true)] {
        let value = json!({
            "model": "m",
            "state": invalid,
            "questions": { "q": { "type": "noul", "instructions": "x" } }
        });
        let report = parse(value).expect_err("invalid state must fail");
        let error = &report.errors()[0];
        assert_eq!(error.kind, error_type::STATE_TYPE, "message: {}", error.msg);
        assert_eq!(
            error.loc,
            vec![LocSegment::key("body"), LocSegment::key("state")]
        );
    }
}

#[test]
fn content_rejects_top_level_number_and_bool() {
    for invalid in [json!(1), json!(true)] {
        let value = json!({
            "model": "m",
            "state": "s",
            "questions": { "q": { "type": "noul", "instructions": invalid } }
        });
        let report = parse(value).expect_err("invalid content must fail");
        let error = &report.errors()[0];
        assert_eq!(error.kind, error_type::JSON_TYPE, "message: {}", error.msg);
        // Internally tagged enums buffer the question map, so the path stops at
        // the question rather than the offending field. The message still names
        // the invalid content type.
        assert_eq!(
            error.loc,
            vec![
                LocSegment::key("body"),
                LocSegment::key("questions"),
                LocSegment::key("q")
            ],
            "message: {}",
            error.msg
        );
    }
}

#[test]
fn content_allows_null_and_nested_json() {
    let value = json!({
        "model": "m",
        "state": "s",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": null,
                "criteria": {
                    "true": "yes",
                    "false": null
                }
            }
        }
    });
    let request = parse(value).expect("valid content");
    let Question::Noul(noul) = &request.questions["q"] else {
        panic!("expected noul question");
    };
    assert_eq!(noul.instructions, Some(Content::Null));
    assert_eq!(
        noul.criteria
            .as_ref()
            .and_then(|criteria| criteria.is_false.clone()),
        Some(Content::Null)
    );
}

#[test]
fn choice_criteria_limits() {
    let empty = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "choice", "criteria": {} } }
    });
    let report = parse(empty).expect_err("empty choice criteria must fail");
    assert_eq!(report.errors()[0].kind, error_type::TOO_SHORT);

    let mut criteria = Map::new();
    for index in 0..256 {
        criteria.insert(format!("c{index}"), json!("x"));
    }
    let too_many = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "choice", "criteria": criteria } }
    });
    let report = parse(too_many).expect_err("256 choice options must fail");
    assert_eq!(report.errors()[0].kind, error_type::TOO_LONG);

    let one = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "choice", "criteria": { "only": "x" } } }
    });
    assert!(parse(one).is_ok());
}

#[test]
fn score_criteria_limits() {
    let one = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "score", "criteria": ["one"] } }
    });
    assert!(parse(one).is_ok(), "official schema allows a single level");

    let eleven = json!({
        "model": "m",
        "state": "s",
        "questions": {
            "q": { "type": "score", "criteria": ["0","1","2","3","4","5","6","7","8","9","10"] }
        }
    });
    let report = parse(eleven).expect_err("11 score levels must fail");
    assert_eq!(report.errors()[0].kind, error_type::TOO_LONG);
}

#[test]
fn empty_questions_and_assets_are_rejected() {
    let empty_questions = json!({ "model": "m", "state": "s", "questions": {} });
    let report = parse(empty_questions).expect_err("empty questions must fail");
    assert_eq!(report.errors()[0].kind, error_type::TOO_SHORT);

    let empty_assets = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "noul", "instructions": "x" } },
        "assets": {}
    });
    let report = parse(empty_assets).expect_err("empty assets must fail");
    assert_eq!(report.errors()[0].kind, error_type::TOO_SHORT);
}

#[test]
fn asset_requires_part_or_inline_data() {
    let value = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "box", "instructions": "x" } },
        "assets": { "photo": { "type": "image" } }
    });
    let report = parse(value).expect_err("asset without data must fail");
    assert_eq!(report.errors()[0].kind, error_type::VALUE_ERROR);
}

#[test]
fn unknown_question_type_is_rejected() {
    let value = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "mystery", "instructions": "x" } }
    });
    let report = parse(value).expect_err("unknown question type must fail");
    assert_eq!(report.errors()[0].kind, error_type::UNION_TAG_INVALID);
}

#[test]
fn missing_model_reports_the_field() {
    let value = json!({
        "state": "s",
        "questions": { "q": { "type": "noul", "instructions": "x" } }
    });
    let report = parse(value).expect_err("missing model must fail");
    let error = &report.errors()[0];
    assert_eq!(error.kind, error_type::MISSING);
    assert_eq!(
        error.loc,
        vec![LocSegment::key("body"), LocSegment::key("model")]
    );
}

#[test]
fn error_response_shapes() {
    let validation = ErrorResponse::validation(vec![ValidationError::new(
        [LocSegment::key("body"), LocSegment::key("state")],
        "Field required",
        error_type::MISSING,
    )]);
    assert_eq!(
        serde_json::to_value(&validation).unwrap(),
        json!({
            "detail": [{
                "loc": ["body", "state"],
                "msg": "Field required",
                "type": "missing"
            }]
        })
    );

    let bad_request = ErrorResponse::bad_request("invalid request");
    assert_eq!(
        serde_json::to_value(&bad_request).unwrap(),
        json!({ "detail": "invalid request" })
    );

    let mut object = Map::new();
    object.insert("message".into(), json!("invalid request"));
    let bad_request_object = ErrorResponse::bad_request_object(object);
    assert_eq!(
        serde_json::to_value(&bad_request_object).unwrap(),
        json!({ "detail": { "message": "invalid request" } })
    );
}

#[test]
fn usage_requires_both_token_counts() {
    let missing_output = json!({
        "model": "m",
        "answers": { "q": { "type": "noul", "noul": 0.5 } },
        "usage": { "input_tokens": 1 }
    });
    assert!(serde_json::from_value::<SystemOneResponse>(missing_output).is_err());
}

#[test]
fn absent_optional_fields_are_omitted() {
    let value = json!({
        "model": "m",
        "state": "s",
        "questions": { "q": { "type": "noul" } }
    });
    let request = parse(value.clone()).expect("valid request");
    assert_eq!(serde_json::to_value(&request).unwrap(), value);
}

#[test]
fn release_date_shape_is_validated() {
    assert!(is_valid_release_date("2026-09-15"));
    assert!(!is_valid_release_date("2026-9-15"));
    assert!(!is_valid_release_date("not-a-date"));
}
