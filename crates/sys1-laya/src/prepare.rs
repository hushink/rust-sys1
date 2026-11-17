use crate::calibration::TYPES;
use crate::config::LayaConfig;
use crate::tokenizer::LayaTokenizer;
use serde_json::Value;
use std::collections::HashMap;
use sys1_core::{ModelError, Questions};
use sys1_protocol::{Content, NoulCriteria, Question, State};

pub(crate) struct PreparedQuestion {
    pub id: String,
    pub kind: usize,
    pub instructions: String,
    pub criteria: Criteria,
    pub labels: Vec<String>,
    pub options: Vec<String>,
}

pub(crate) enum Criteria {
    Choice,
    Score(Vec<Content>),
    Noul,
}

pub(crate) struct Item {
    pub question: PreparedQuestion,
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

pub(crate) struct PreparedRequest {
    pub items: Vec<Item>,
}

pub(crate) fn prepare_request(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    state: &State,
    questions: &Questions,
) -> Result<PreparedRequest, ModelError> {
    let state_ids = state_token_ids(tokenizer, state)?;
    prepare_request_with_state_ids(tokenizer, config, &state_ids, questions)
}

/// Tokenize a state once, before it is split into long-context windows.
pub(crate) fn state_token_ids(
    tokenizer: &LayaTokenizer,
    state: &State,
) -> Result<Vec<u32>, ModelError> {
    let state_text = sanitize_masks(&render_state(state));
    tokenizer.encode(&state_text).map_err(ModelError::inference)
}

/// Prepare a request from state token ids that were already tokenized (and,
/// for long-context scanning, sliced into windows).
pub(crate) fn prepare_request_with_state_ids(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    state_ids: &[u32],
    questions: &Questions,
) -> Result<PreparedRequest, ModelError> {
    if questions.is_empty() {
        return Err(ModelError::invalid_request("questions must not be empty"));
    }

    let mut items = Vec::with_capacity(questions.len());
    for (id, question) in questions.iter() {
        let prepared = prepare_question(id, question)?;
        let (ids, markers) = build_sequence(tokenizer, config, state_ids, &prepared)?;
        items.push(Item {
            question: prepared,
            ids,
            markers,
        });
    }
    Ok(PreparedRequest { items })
}

fn prepare_question(id: &str, question: &Question) -> Result<PreparedQuestion, ModelError> {
    match question {
        Question::Choice(choice) => {
            let instructions = instructions_or_id(choice.instructions.as_ref(), id);
            let labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
            let options = choice
                .criteria
                .iter()
                .map(|(label, description)| option_text(label, description))
                .collect::<Vec<_>>();
            Ok(PreparedQuestion {
                id: id.to_owned(),
                kind: 0,
                instructions,
                criteria: Criteria::Choice,
                labels,
                options,
            })
        }
        Question::Score(score) => {
            if score.criteria.is_empty() {
                return Err(ModelError::invalid_request(format!(
                    "score question {id:?} requires criteria"
                )));
            }
            if score.criteria.len() > 10 {
                return Err(ModelError::invalid_request(format!(
                    "score question {id:?} supports at most 10 criteria"
                )));
            }
            let instructions = instructions_or_id(score.instructions.as_ref(), id);
            let labels = (0..score.criteria.len())
                .map(|index| index.to_string())
                .collect::<Vec<_>>();
            let options = score
                .criteria
                .iter()
                .enumerate()
                .map(|(index, value)| format!("level {index}: {}", render_content(value)))
                .collect::<Vec<_>>();
            Ok(PreparedQuestion {
                id: id.to_owned(),
                kind: 1,
                instructions,
                criteria: Criteria::Score(score.criteria.clone()),
                labels,
                options,
            })
        }
        Question::Noul(noul) => {
            let instructions = instructions_or_id(noul.instructions.as_ref(), id);
            let (false_text, true_text) = noul_texts(noul.criteria.as_ref());
            Ok(PreparedQuestion {
                id: id.to_owned(),
                kind: 2,
                instructions,
                criteria: Criteria::Noul,
                labels: vec!["false".to_owned(), "true".to_owned()],
                options: vec![false_text, true_text],
            })
        }
        Question::Box(_) => Err(ModelError::unsupported(format!(
            "question {id:?}: Laya does not support box questions"
        ))),
    }
}

/// How many state tokens a question leaves inside `max_len`.
pub(crate) fn state_room(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    id: &str,
    question: &Question,
) -> Result<usize, ModelError> {
    let prepared = prepare_question(id, question)?;
    let (head, _) = build_head(tokenizer, config, &prepared)?;
    Ok(config.max_len.saturating_sub(head.len() + 1))
}

fn build_head(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    question: &PreparedQuestion,
) -> Result<(Vec<u32>, Vec<usize>), ModelError> {
    let mut head = tokenizer
        .encode(&format!(
            "{} question: {}",
            TYPES[question.kind],
            sanitize_masks(&question.instructions)
        ))
        .map_err(ModelError::inference)?;

    let mut options = Vec::with_capacity(question.options.len());
    for option in &question.options {
        let mut ids = vec![tokenizer.mask_id];
        ids.extend(
            tokenizer
                .encode(&format!(" {}", sanitize_masks(option)))
                .map_err(ModelError::inference)?
                .into_iter()
                .take(48),
        );
        options.push(ids);
    }

    let mut budget =
        config.head_max_len as isize - options.iter().map(Vec::len).sum::<usize>() as isize;
    if budget < 16 {
        let per = ((config.head_max_len.saturating_sub(16)) / options.len().max(1)).max(4);
        options.iter_mut().for_each(|option| option.truncate(per));
        budget =
            config.head_max_len as isize - options.iter().map(Vec::len).sum::<usize>() as isize;
    }
    head.truncate(budget.max(8) as usize);

    let mut ids = vec![tokenizer.cls_id];
    ids.extend(head);
    ids.push(tokenizer.sep_id);

    let mut markers = Vec::with_capacity(options.len());
    for option in options {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(tokenizer.sep_id);
    Ok((ids, markers))
}

fn build_sequence(
    tokenizer: &LayaTokenizer,
    config: &LayaConfig,
    state_ids: &[u32],
    question: &PreparedQuestion,
) -> Result<(Vec<u32>, Vec<usize>), ModelError> {
    let (mut ids, mut markers) = build_head(tokenizer, config, question)?;

    let room = config.max_len.saturating_sub(ids.len() + 1);
    if state_ids.len() > room {
        return Err(ModelError::context_length_exceeded(format!(
            "max_tokens_exceeded: state has {} tokens but only {room} fit in max_len={}",
            state_ids.len(),
            config.max_len
        )));
    }
    ids.extend(state_ids.iter().copied());
    ids.push(tokenizer.sep_id);
    ids.truncate(config.max_len);
    markers.retain(|marker| *marker < config.max_len);

    if markers.len() != question.options.len() {
        return Err(ModelError::invalid_request(format!(
            "question {:?} options exceed head_max_len={}",
            question.id, config.head_max_len
        )));
    }

    Ok((ids, markers))
}

fn instructions_or_id(instructions: Option<&Content>, id: &str) -> String {
    match instructions {
        Some(Content::Null) | None => id.to_owned(),
        Some(Content::String(value)) if value.is_empty() => id.to_owned(),
        Some(content) => render_content(content),
    }
}

fn option_text(label: &str, description: &Content) -> String {
    match description {
        Content::Null => label.to_owned(),
        Content::String(value) if value.is_empty() => label.to_owned(),
        other => format!("{label}: {}", render_content(other)),
    }
}

fn noul_texts(criteria: Option<&NoulCriteria>) -> (String, String) {
    let false_value = criteria.and_then(|criteria| criteria.is_false.as_ref());
    let true_value = criteria.and_then(|criteria| criteria.is_true.as_ref());
    (
        format!(
            "false: {}",
            criterion_text(false_value, "no, the statement does not hold")
        ),
        format!(
            "true: {}",
            criterion_text(true_value, "yes, the statement holds")
        ),
    )
}

fn criterion_text(value: Option<&Content>, default: &str) -> String {
    match value {
        Some(Content::Null) | None => default.to_owned(),
        Some(Content::String(value)) if value.is_empty() => default.to_owned(),
        Some(content) => render_content(content),
    }
}

pub(crate) fn render_state(state: &State) -> String {
    render_content(state.as_content())
}

fn render_content(content: &Content) -> String {
    match content {
        Content::String(value) => value.clone(),
        other => render_json(&content_to_value(other)),
    }
}

fn content_to_value(content: &Content) -> Value {
    match content {
        Content::Null => Value::Null,
        Content::String(value) => Value::String(value.clone()),
        Content::Object(value) => Value::Object(value.clone()),
        Content::Array(value) => Value::Array(value.clone()),
    }
}

struct ItemKey<'a> {
    kind: usize,
    ids: &'a [u32],
    markers: &'a [usize],
}

impl PartialEq for ItemKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.ids == other.ids && self.markers == other.markers
    }
}

impl Eq for ItemKey<'_> {}

impl std::hash::Hash for ItemKey<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.ids.hash(state);
        self.markers.hash(state);
    }
}

/// Collapse identical token sequences before a forward pass. Returns the
/// unique items in first-seen order and one index per input item.
pub(crate) fn deduplicate<'a>(items: &[&'a Item]) -> (Vec<&'a Item>, Vec<usize>) {
    let mut unique = Vec::with_capacity(items.len());
    let mut indices = Vec::with_capacity(items.len());
    let mut seen: HashMap<ItemKey<'a>, usize> = HashMap::with_capacity(items.len());
    for item in items {
        let key = ItemKey {
            kind: item.question.kind,
            ids: &item.ids,
            markers: &item.markers,
        };
        let index = *seen.entry(key).or_insert_with(|| {
            let index = unique.len();
            unique.push(*item);
            index
        });
        indices.push(index);
    }
    (unique, indices)
}

fn render_json(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).unwrap(),
                    render_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        value => serde_json::to_string(value).unwrap(),
    }
}

pub(crate) fn sanitize_masks(value: &str) -> String {
    value.replace("[MASK]", " ").replace("<mask>", " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use serde_json::{Map, json};
    use std::env;
    use std::fs;
    use sys1_protocol::Question;

    #[test]
    #[ignore = "requires a cached Laya checkpoint and official golden vectors"]
    fn matches_multilingual_golden_ids() {
        let root = env::var("SYS1_LAYA_TEST_ROOT")
            .map(std::path::PathBuf::from)
            .expect("set SYS1_LAYA_TEST_ROOT");
        let golden = env::var("SYS1_LAYA_GOLDEN_DIR")
            .map(std::path::PathBuf::from)
            .expect("set SYS1_LAYA_GOLDEN_DIR");
        let directory = root.join("multilingual");
        let config = LayaConfig::load(&directory.join("rl_agent_config.json")).expect("config");
        let tokenizer = LayaTokenizer::load(&directory).expect("tokenizer");
        let case: Value = serde_json::from_slice(
            &fs::read(golden.join("multilingual").join("case_dict_state.json"))
                .expect("golden case"),
        )
        .expect("golden JSON");
        let state = State::from_value(case["state"].clone()).expect("state");
        let questions: IndexMap<String, Question> =
            serde_json::from_value(case["questions"].clone()).expect("questions");
        let prepared = prepare_request(&tokenizer, &config, &state, &Questions::from(questions))
            .expect("prepare");
        let expected_ids: Vec<u32> =
            serde_json::from_value(case["per_question"][0]["input_ids"].clone()).expect("ids");
        let expected_markers: Vec<usize> =
            serde_json::from_value(case["per_question"][0]["markers"].clone()).expect("markers");
        assert_eq!(prepared.items[0].ids, expected_ids, "input_ids");
        assert_eq!(prepared.items[0].markers, expected_markers, "markers");
    }

    #[test]
    fn renders_python_style_json() {
        assert_eq!(
            render_json(&json!({"a": 1, "b": [true, null]})),
            "{\"a\": 1, \"b\": [true, null]}"
        );
        assert_eq!(render_json(&json!(["a", "b"])), "[\"a\", \"b\"]");
    }

    #[test]
    fn renders_options_and_instructions() {
        assert_eq!(option_text("calm", &Content::Null), "calm");
        assert_eq!(
            option_text("calm", &Content::String("neutral".to_owned())),
            "calm: neutral"
        );
        assert_eq!(
            instructions_or_id(Some(&Content::Null), "billing"),
            "billing"
        );
        assert_eq!(
            instructions_or_id(Some(&Content::String("Is this billing?".to_owned())), "x"),
            "Is this billing?"
        );
        assert_eq!(
            instructions_or_id(
                Some(&Content::Object(Map::from_iter([(
                    "task".to_owned(),
                    json!("classify")
                )]))),
                "x"
            ),
            "{\"task\": \"classify\"}"
        );
    }

    #[test]
    fn sanitizes_mask_tokens() {
        assert_eq!(sanitize_masks("a [MASK] b <mask> c"), "a   b   c");
    }

    #[test]
    fn deduplicates_identical_inputs_without_losing_order() {
        let make = |id: &str, kind: usize, ids: &[u32]| Item {
            question: PreparedQuestion {
                id: id.to_owned(),
                kind,
                instructions: String::new(),
                criteria: Criteria::Noul,
                labels: Vec::new(),
                options: Vec::new(),
            },
            ids: ids.to_vec(),
            markers: vec![1],
        };
        let first = make("first", 0, &[1, 2, 3]);
        let duplicate = make("duplicate", 0, &[1, 2, 3]);
        let other_kind = make("other-kind", 1, &[1, 2, 3]);
        let items = vec![&first, &duplicate, &other_kind, &first];

        let (unique, indices) = deduplicate(&items);

        assert_eq!(unique.len(), 2);
        assert_eq!(indices, vec![0, 0, 1, 0]);
    }
}
