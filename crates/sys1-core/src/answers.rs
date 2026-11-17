use crate::error::AnswerError;
use indexmap::IndexMap;
use sys1_protocol::{Answer, AnswerType, BoxAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer};

/// Answers keyed by the question id supplied in the request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Answers(IndexMap<String, Answer>);

impl Answers {
    pub fn new(answers: IndexMap<String, Answer>) -> Self {
        Self(answers)
    }

    pub fn get(&self, id: &str) -> Option<&Answer> {
        self.0.get(id)
    }

    pub fn choice(&self, id: &str) -> Result<&ChoiceAnswer, AnswerError> {
        match self.get(id) {
            Some(Answer::Choice(answer)) => Ok(answer),
            Some(other) => Err(AnswerError::WrongType {
                id: id.to_owned(),
                expected: AnswerType::Choice,
                actual: other.answer_type(),
            }),
            None => Err(AnswerError::Missing { id: id.to_owned() }),
        }
    }

    pub fn score(&self, id: &str) -> Result<&ScoreAnswer, AnswerError> {
        match self.get(id) {
            Some(Answer::Score(answer)) => Ok(answer),
            Some(other) => Err(AnswerError::WrongType {
                id: id.to_owned(),
                expected: AnswerType::Score,
                actual: other.answer_type(),
            }),
            None => Err(AnswerError::Missing { id: id.to_owned() }),
        }
    }

    pub fn noul(&self, id: &str) -> Result<&NoulAnswer, AnswerError> {
        match self.get(id) {
            Some(Answer::Noul(answer)) => Ok(answer),
            Some(other) => Err(AnswerError::WrongType {
                id: id.to_owned(),
                expected: AnswerType::Noul,
                actual: other.answer_type(),
            }),
            None => Err(AnswerError::Missing { id: id.to_owned() }),
        }
    }

    pub fn boxed(&self, id: &str) -> Result<&BoxAnswer, AnswerError> {
        match self.get(id) {
            Some(Answer::Box(answer)) => Ok(answer),
            Some(other) => Err(AnswerError::WrongType {
                id: id.to_owned(),
                expected: AnswerType::Box,
                actual: other.answer_type(),
            }),
            None => Err(AnswerError::Missing { id: id.to_owned() }),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Answer)> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_map(&self) -> &IndexMap<String, Answer> {
        &self.0
    }

    pub fn into_map(self) -> IndexMap<String, Answer> {
        self.0
    }
}

impl From<IndexMap<String, Answer>> for Answers {
    fn from(answers: IndexMap<String, Answer>) -> Self {
        Self(answers)
    }
}

impl FromIterator<(String, Answer)> for Answers {
    fn from_iter<T: IntoIterator<Item = (String, Answer)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
