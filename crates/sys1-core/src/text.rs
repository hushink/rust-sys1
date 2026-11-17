//! Python-compatible text rendering shared by model families.
//!
//! Several decision checkpoints were trained on text produced by CPython:
//! `json.dumps` with its default separators, Python `str()`/`repr()` for
//! structured state, and `str.strip()` whitespace rules. Reproducing that text
//! byte for byte keeps token rows aligned with the upstream reference.

use serde_json::{Map, Number, Value};
use std::fmt::Write;
use sys1_protocol::{Content, State};
use unicode_general_category::{GeneralCategory, get_general_category};

/// `json.dumps(value, ensure_ascii=ensure_ascii)` with the default separators.
pub fn python_json(value: &Value, ensure_ascii: bool) -> String {
    dumps(value, ensure_ascii, false)
}

/// `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(&mut out, value);
    out
}

/// `json.dumps` for the JSON projection of a [`Content`] or [`State`].
pub fn content_json(content: &Content, ensure_ascii: bool) -> String {
    python_json(&content_to_value(content), ensure_ascii)
}

fn dumps(value: &Value, ascii: bool, sort_keys: bool) -> String {
    let mut out = String::new();
    write_python_json(&mut out, value, ascii, sort_keys);
    out
}

fn write_python_json(out: &mut String, value: &Value, ascii: bool, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(out, number),
        Value::String(text) => write_json_string(out, text, ascii),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(out, item, ascii, sort_keys);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|left, right| left.0.cmp(right.0));
            }
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_json_string(out, key, ascii);
                out.push_str(": ");
                write_python_json(out, item, ascii, sort_keys);
            }
            out.push('}');
        }
    }
}

fn write_canonical(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(out, number),
        Value::String(text) => write_json_string(out, text, false),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json_string(out, key, false);
                out.push(':');
                write_canonical(out, item);
            }
            out.push('}');
        }
    }
}

fn write_number(out: &mut String, number: &Number) {
    if let Some(value) = number.as_i64() {
        let _ = write!(out, "{value}");
    } else if let Some(value) = number.as_u64() {
        let _ = write!(out, "{value}");
    } else {
        out.push_str(&float_repr(number.as_f64().unwrap_or(f64::NAN)));
    }
}

fn write_json_string(out: &mut String, text: &str, ascii: bool) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character if ascii && (character as u32) > 0x7e => {
                let code = character as u32;
                if code > 0xffff {
                    let code = code - 0x10000;
                    let _ = write!(
                        out,
                        "\\u{:04x}\\u{:04x}",
                        0xd800 | (code >> 10),
                        0xdc00 | (code & 0x3ff)
                    );
                } else {
                    let _ = write!(out, "\\u{code:04x}");
                }
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

/// Python's `repr(float)` and `json.dumps` float formatting.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("LowerExp exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let decimal_point = exponent + 1;

    let mut out = String::new();
    if value < 0.0 {
        out.push('-');
    }
    if decimal_point <= -4 || decimal_point > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    } else if decimal_point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decimal_point) as usize));
        out.push_str(&digits);
    } else if decimal_point as usize >= digits.len() {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n(
            '0',
            decimal_point as usize - digits.len(),
        ));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decimal_point as usize]);
        out.push('.');
        out.push_str(&digits[decimal_point as usize..]);
    }
    out
}

/// Python `str(value)`: a top-level string is verbatim, everything else uses
/// `repr`.
pub fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        value => python_repr(value),
    }
}

/// Python `repr(value)` for a value produced by `json.loads`.
pub fn python_repr(value: &Value) -> String {
    let mut out = String::new();
    write_repr(&mut out, value);
    out
}

fn write_repr(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(number) => write_number(out, number),
        Value::String(text) => write_repr_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_repr(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_repr_string(out, key);
                out.push_str(": ");
                write_repr(out, item);
            }
            out.push('}');
        }
    }
}

fn write_repr_string(out: &mut String, text: &str) {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for character in text.chars() {
        match character {
            character if character == quote || character == '\\' => {
                out.push('\\');
                out.push(character);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            character if character < ' ' || character == '\u{7f}' => {
                let _ = write!(out, "\\x{:02x}", character as u32);
            }
            character if character.is_ascii() || is_printable(character) => out.push(character),
            character => {
                let code = character as u32;
                let _ = if code <= 0xff {
                    write!(out, "\\x{code:02x}")
                } else if code <= 0xffff {
                    write!(out, "\\u{code:04x}")
                } else {
                    write!(out, "\\U{code:08x}")
                };
            }
        }
    }
    out.push(quote);
}

/// `str.isprintable`: separators other than U+0020, controls, formats,
/// surrogates, private-use and unassigned characters are not printable.
pub fn is_printable(character: char) -> bool {
    if character == ' ' {
        return true;
    }
    !matches!(
        get_general_category(character),
        GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::Surrogate
            | GeneralCategory::PrivateUse
            | GeneralCategory::Unassigned
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
            | GeneralCategory::SpaceSeparator
    )
}

/// Python `bool(value)` for a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python `str.strip()`, which also strips U+001C..U+001F.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_python_space)
}

/// Python `str.split()` without arguments.
pub fn split(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_python_space).filter(|word| !word.is_empty())
}

pub fn is_python_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// Render a [`State`] the way the decision layouts read it: strings verbatim,
/// objects and arrays as `json.dumps(ensure_ascii=False)`.
pub fn render_state(state: &State) -> String {
    render_content(state.as_content())
}

/// Render a [`Content`] value the same way as [`render_state`].
pub fn render_content(content: &Content) -> String {
    match content {
        Content::String(text) => text.clone(),
        content => content_json(content, false),
    }
}

/// Render a question's instructions, falling back to its id when absent or
/// null.
pub fn render_instructions(instructions: Option<&Content>, id: &str) -> String {
    match instructions {
        None | Some(Content::Null) => id.to_owned(),
        Some(content) => render_content(content),
    }
}

/// Render a criterion: null as an empty string, strings verbatim, structured
/// values as JSON.
pub fn render_criterion(content: &Content) -> String {
    match content {
        Content::Null => String::new(),
        content => render_content(content),
    }
}

pub fn content_to_value(content: &Content) -> Value {
    match content {
        Content::Null => Value::Null,
        Content::String(text) => Value::String(text.clone()),
        Content::Object(map) => Value::Object(map.clone()),
        Content::Array(items) => Value::Array(items.clone()),
    }
}

/// Render a value as `key: value` lines, the textual form several layouts use
/// for a JSON object state.
pub fn render_object_lines(value: &Value) -> String {
    match value {
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| format!("{key}: {}", python_str(value)))
            .collect::<Vec<_>>()
            .join("\n"),
        value => python_str(value),
    }
}

/// Replace a template's `{name}` placeholders in one left-to-right pass.
pub fn fill_template(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len() + 32);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let end = after
            .find(|character: char| !(character.is_alphanumeric() || character == '_'))
            .unwrap_or(after.len());
        if end > 0
            && after[end..].starts_with('}')
            && let Some(value) = values
                .iter()
                .find_map(|(key, value)| (*key == &after[..end]).then_some(*value))
        {
            out.push_str(value);
            rest = &after[end + 1..];
            continue;
        }
        out.push('{');
        rest = after;
    }
    out.push_str(rest);
    out
}

/// A JSON object with keys in code-point order, for canonical rendering.
pub fn sorted_object(entries: impl IntoIterator<Item = (String, Value)>) -> Map<String, Value> {
    let mut entries: Vec<(String, Value)> = entries.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    entries.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_match_python_repr() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (19.99, "19.99"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (123.456, "123.456"),
            (-0.0, "-0.0"),
            (0.1 + 0.2, "0.30000000000000004"),
        ];
        for (value, expected) in cases {
            assert_eq!(float_repr(*value), *expected);
        }
    }

    #[test]
    fn json_matches_python() {
        let value = json!({"a": [1, 2.0, true, null], "é": "x\"y\\z\n\u{1}", "emoji": "😡"});
        assert_eq!(
            python_json(&value, false),
            "{\"a\": [1, 2.0, true, null], \"é\": \"x\\\"y\\\\z\\n\\u0001\", \"emoji\": \"😡\"}"
        );
        assert_eq!(
            canonical_json(
                &json!({"key": "b", "description": {"z": [1, 2.5, null], "é": "x\n", "A": true, "a": 1e-5}})
            ),
            r#"{"description":{"A":true,"a":1e-05,"z":[1,2.5,null],"é":"x\n"},"key":"b"}"#
        );
    }

    #[test]
    fn repr_matches_python() {
        let value = json!({"k": "v", "n": 1, "f": 19.99, "e": 1e-5, "b": true, "x": null,
                           "l": ["a", 2.0, false, [], {}]});
        assert_eq!(
            python_repr(&value),
            "{'k': 'v', 'n': 1, 'f': 19.99, 'e': 1e-05, 'b': True, 'x': None, 'l': ['a', 2.0, False, [], {}]}"
        );
        assert_eq!(python_repr(&json!("it's")), "\"it's\"");
        assert_eq!(python_repr(&json!("\u{a0}\u{ad}é")), "'\\xa0\\xadé'");
        assert_eq!(python_repr(&json!("👨\u{200d}👩 ok")), "'👨\\u200d👩 ok'");
        assert_eq!(python_str(&json!(["it's"])), "[\"it's\"]");
    }

    #[test]
    fn whitespace_matches_python() {
        assert_eq!(strip("\u{1c}\u{a0} x y \u{3000}\n"), "x y");
        assert_eq!(
            split("\u{1f}a\tb\u{a0}c\u{200b}d  \u{3000}e\u{85}f\u{2028} ").collect::<Vec<_>>(),
            ["a", "b", "c\u{200b}d", "e", "f"]
        );
        assert!(truthy(&json!({"a": 0})));
        assert!(!truthy(&json!([])));
    }

    #[test]
    fn fills_templates_in_one_pass() {
        assert_eq!(
            fill_template("{a} {b} {a}", &[("a", "{b}"), ("b", "x")]),
            "{b} x {b}"
        );
        assert_eq!(fill_template("no fields", &[]), "no fields");
    }
}
