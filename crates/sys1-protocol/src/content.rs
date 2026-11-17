use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use thiserror::Error;

/// The official `Content` union: a string, object, array, or null.
///
/// Top-level numbers and booleans are rejected. Inside objects and arrays any
/// JSON value is allowed.
#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    Null,
    String(String),
    Object(Map<String, Value>),
    Array(Vec<Value>),
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ContentError {
    #[error("content must be a string, object, array, or null")]
    InvalidType,
}

impl Content {
    pub fn from_value(value: Value) -> Result<Self, ContentError> {
        match value {
            Value::Null => Ok(Self::Null),
            Value::String(value) => Ok(Self::String(value)),
            Value::Object(value) => Ok(Self::Object(value)),
            Value::Array(value) => Ok(Self::Array(value)),
            Value::Number(_) | Value::Bool(_) => Err(ContentError::InvalidType),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }
}

impl From<&str> for Content {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<String> for Content {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl TryFrom<Value> for Content {
    type Error = ContentError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Self::from_value(value)
    }
}

impl Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::String(value) => serializer.serialize_str(value),
            Self::Object(value) => value.serialize(serializer),
            Self::Array(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(value).map_err(serde::de::Error::custom)
    }
}

/// Preserve explicit JSON `null` as `Some(Content::Null)` instead of `None`.
///
/// `Option<T>` normally maps JSON null to `None`, which loses the distinction
/// between an absent field and an explicitly null one. The official contract
/// allows explicit null in `instructions` and `criteria`.
pub(crate) fn deserialize_optional_content<'de, D>(
    deserializer: D,
) -> Result<Option<Content>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Content::from_value(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
}
