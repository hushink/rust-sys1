use crate::content::Content;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use thiserror::Error;

/// The content all questions in a request refer to.
///
/// The official schema allows a string, object, or array. Null, numbers, and
/// booleans are rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct State(Content);

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StateError {
    #[error("state must be a string, object, or array")]
    InvalidType,
    #[error(transparent)]
    Content(#[from] crate::content::ContentError),
}

impl State {
    pub fn text(value: impl Into<String>) -> Self {
        Self(Content::String(value.into()))
    }

    pub fn object(value: Map<String, Value>) -> Self {
        Self(Content::Object(value))
    }

    pub fn array(value: Vec<Value>) -> Self {
        Self(Content::Array(value))
    }

    pub fn from_value(value: Value) -> Result<Self, StateError> {
        match value {
            Value::String(_) | Value::Object(_) | Value::Array(_) => {
                Ok(Self(Content::from_value(value)?))
            }
            Value::Null | Value::Number(_) | Value::Bool(_) => Err(StateError::InvalidType),
        }
    }

    pub fn as_content(&self) -> &Content {
        &self.0
    }

    pub fn into_content(self) -> Content {
        self.0
    }

    pub fn as_str(&self) -> Option<&str> {
        self.0.as_str()
    }

    pub fn as_object(&self) -> Option<&Map<String, Value>> {
        self.0.as_object()
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        self.0.as_array()
    }
}

impl From<&str> for State {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

impl From<String> for State {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl TryFrom<Value> for State {
    type Error = StateError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Self::from_value(value)
    }
}

impl Serialize for State {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for State {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(value).map_err(serde::de::Error::custom)
    }
}
