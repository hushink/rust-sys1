use indexmap::IndexMap;
use sys1_protocol::{
    BoxQuestion, ChoiceQuestion, Content, NoulQuestion, Question, ScoreQuestion, ValidationReport,
    validate_questions,
};

/// Questions keyed by a caller-chosen id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Questions(IndexMap<String, Question>);

impl Questions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(IndexMap::with_capacity(capacity))
    }

    pub fn insert(mut self, id: impl Into<String>, question: Question) -> Self {
        self.0.insert(id.into(), question);
        self
    }

    pub fn choice<K, V, I>(mut self, id: impl Into<String>, criteria: I) -> Self
    where
        K: Into<String>,
        V: Into<Content>,
        I: IntoIterator<Item = (K, V)>,
    {
        let criteria = criteria
            .into_iter()
            .map(|(label, description)| (label.into(), description.into()))
            .collect();
        self.0
            .insert(id.into(), Question::Choice(ChoiceQuestion::new(criteria)));
        self
    }

    pub fn score<V, I>(mut self, id: impl Into<String>, criteria: I) -> Self
    where
        V: Into<Content>,
        I: IntoIterator<Item = V>,
    {
        let criteria: Vec<Content> = criteria.into_iter().map(Into::into).collect();
        self.0
            .insert(id.into(), Question::Score(ScoreQuestion::new(criteria)));
        self
    }

    pub fn noul(mut self, id: impl Into<String>, instructions: impl Into<Content>) -> Self {
        self.0
            .insert(id.into(), Question::Noul(NoulQuestion::new(instructions)));
        self
    }

    pub fn boxed(mut self, id: impl Into<String>, instructions: impl Into<Content>) -> Self {
        self.0
            .insert(id.into(), Question::Box(BoxQuestion::new(instructions)));
        self
    }

    pub fn get(&self, id: &str) -> Option<&Question> {
        self.0.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Question)> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_map(&self) -> &IndexMap<String, Question> {
        &self.0
    }

    pub fn into_map(self) -> IndexMap<String, Question> {
        self.0
    }

    pub fn validate(&self) -> Result<(), ValidationReport> {
        validate_questions(&self.0)
    }
}

impl From<IndexMap<String, Question>> for Questions {
    fn from(questions: IndexMap<String, Question>) -> Self {
        Self(questions)
    }
}

impl From<Questions> for IndexMap<String, Question> {
    fn from(questions: Questions) -> Self {
        questions.0
    }
}

impl FromIterator<(String, Question)> for Questions {
    fn from_iter<T: IntoIterator<Item = (String, Question)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
