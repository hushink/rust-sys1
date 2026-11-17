use serde_json::Value;
use sys1_protocol::{Content, State};

pub(crate) fn render_state(state: &State) -> String {
    render_content(state.as_content())
}

pub(crate) fn render_content(content: &Content) -> String {
    match content {
        Content::String(value) => value.clone(),
        other => render_json(&content_to_value(other)),
    }
}

pub(crate) fn instructions_or_id(instructions: Option<&Content>, id: &str) -> String {
    match instructions {
        Some(Content::Null) | None => id.to_owned(),
        Some(Content::String(value)) if value.is_empty() => id.to_owned(),
        Some(content) => render_content(content),
    }
}

pub(crate) fn criterion(content: &Content) -> String {
    match content {
        Content::Null => String::new(),
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

pub(crate) fn render_json(value: &Value) -> String {
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
                    serde_json::to_string(key).expect("JSON key"),
                    render_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        value => serde_json::to_string(value).expect("JSON value"),
    }
}

/// Replace GLiClass's structural markers so user text cannot create labels or
/// segments.
pub(crate) fn sanitize(text: &str) -> String {
    text.replace("<<LABEL>>", " ")
        .replace("<<SEP>>", " ")
        .replace("<<EXAMPLE>>", " ")
}
