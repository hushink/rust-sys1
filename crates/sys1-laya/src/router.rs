use crate::language::{self, LanguageAnalysis};
use crate::model::{LayaModel, LayaOptions};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use sys1_core::{DecisionModel, DecisionOutput, ModelError, Prediction, Questions};
use sys1_protocol::State;
use unicode_general_category::{GeneralCategory, get_general_category};

/// One of the three published Laya checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RouteVariant {
    English,
    Multilingual,
    TypedDecisions,
}

impl RouteVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }

    /// Parse a checkpoint name or alias; case-insensitive.
    pub fn from_name(name: &str) -> Option<Self> {
        let key = name.trim().to_ascii_lowercase();
        match key.as_str() {
            "en" | "laya" | "default" | "english" => Some(Self::English),
            "multi" | "ml" | "laya-multilingual" | "multilingual" => Some(Self::Multilingual),
            "typed"
            | "typed_decisions"
            | "typed-decisions"
            | "laya-typed-decisions"
            | "decisions" => Some(Self::TypedDecisions),
            _ => None,
        }
    }
}

/// Router-wide settings.
#[derive(Clone, Copy, Debug)]
pub struct LayaRouterOptions {
    /// Checkpoint for text whose language detection abstains.
    pub default: RouteVariant,
    /// Opt in to selecting `typed-decisions` when the question ids match one
    /// of the four published workflows. Official Laya defaults this to false.
    pub auto_task_detection: bool,
}

impl Default for LayaRouterOptions {
    fn default() -> Self {
        Self {
            default: RouteVariant::Multilingual,
            auto_task_detection: false,
        }
    }
}

/// Per-call routing overrides, mirroring the official `Router.route` arguments.
#[derive(Clone, Debug, Default)]
pub struct RouteHints {
    pub model: Option<String>,
    pub task: Option<String>,
    pub lang: Option<String>,
    pub lang_guess: Option<String>,
}

/// The routing outcome, including the detected language evidence.
#[derive(Clone, Debug)]
pub struct RouteDecision {
    pub model: RouteVariant,
    pub reason: String,
    pub detection: Option<LanguageAnalysis>,
    pub workflow: Option<String>,
}

impl RouteDecision {
    pub fn model_name(&self) -> &'static str {
        self.model.as_str()
    }
}

/// Laya's automatic router across the three published checkpoints.
///
/// It mirrors the official routing ladder: explicit model/task, optional
/// typed-decisions workflow detection, explicit language hints, then script
/// and Latin-language detection. The default checkpoint is `multilingual`;
/// `typed-decisions` is never selected automatically unless
/// [`LayaRouterOptions::auto_task_detection`] is enabled.
pub struct LayaRouter {
    english: PathBuf,
    multilingual: PathBuf,
    typed_decisions: PathBuf,
    options: LayaOptions,
    routing: LayaRouterOptions,
    lang_guess: Option<String>,
    loaded: Mutex<IndexMap<RouteVariant, Arc<LayaModel>>>,
    max_loaded: usize,
}

impl LayaRouter {
    /// `root` is the Laya repository root: it must contain `multilingual/` and
    /// `typed-decisions/` subdirectories beside the English checkpoint.
    pub fn new(root: impl Into<PathBuf>, options: LayaOptions) -> Self {
        let root = root.into();
        Self::with_paths(
            root.clone(),
            root.join("multilingual"),
            root.join("typed-decisions"),
            options,
        )
    }

    pub fn with_paths(
        english: impl Into<PathBuf>,
        multilingual: impl Into<PathBuf>,
        typed_decisions: impl Into<PathBuf>,
        options: LayaOptions,
    ) -> Self {
        Self {
            english: english.into(),
            multilingual: multilingual.into(),
            typed_decisions: typed_decisions.into(),
            routing: options.router,
            lang_guess: options.lang_guess.clone(),
            options,
            loaded: Mutex::new(IndexMap::new()),
            max_loaded: 2,
        }
    }

    /// Replace router-wide routing settings.
    pub fn set_routing(&mut self, routing: LayaRouterOptions) {
        self.routing = routing;
    }

    /// Install a router-wide language hint (a code such as `en` or `ro`).
    pub fn set_lang_guess(&mut self, lang_guess: Option<String>) {
        self.lang_guess = lang_guess;
    }

    /// Route with the router's default settings.
    pub fn route(&self, state: &State, questions: &Questions) -> Result<RouteDecision, ModelError> {
        self.route_with(state, questions, &RouteHints::default())
    }

    /// Route with per-call overrides.
    pub fn route_with(
        &self,
        state: &State,
        questions: &Questions,
        hints: &RouteHints,
    ) -> Result<RouteDecision, ModelError> {
        if let Some(model) = hints.model.as_deref() {
            let variant = RouteVariant::from_name(model).ok_or_else(|| {
                ModelError::invalid_request(format!("unknown routing model {model:?}"))
            })?;
            return Ok(RouteDecision {
                model: variant,
                reason: format!("explicit model={}", python_repr(model)),
                detection: None,
                workflow: None,
            });
        }
        if let Some(task) = hints.task.as_deref() {
            let key = task.trim().to_ascii_lowercase().replace('-', "_");
            let variant = if key == "typed_decisions" {
                RouteVariant::TypedDecisions
            } else {
                RouteVariant::from_name(task).ok_or_else(|| {
                    ModelError::invalid_request(format!("unknown routing task {task:?}"))
                })?
            };
            return Ok(RouteDecision {
                model: variant,
                reason: format!("explicit task={}", python_repr(task)),
                detection: None,
                workflow: None,
            });
        }

        let workflow = match_typed_decisions_workflow(questions);
        if self.routing.auto_task_detection
            && let Some(workflow) = workflow.as_deref()
        {
            return Ok(RouteDecision {
                model: RouteVariant::TypedDecisions,
                reason: format!("question ids match the '{workflow}' typed-decisions workflow"),
                detection: None,
                workflow: Some(workflow.to_owned()),
            });
        }

        if let Some(lang) = hints.lang.as_deref()
            && let Some(english) = english_from_code(lang)
        {
            return Ok(RouteDecision {
                model: if english {
                    RouteVariant::English
                } else {
                    RouteVariant::Multilingual
                },
                reason: format!("explicit lang={}", python_repr(lang)),
                detection: None,
                workflow: workflow.clone(),
            });
        }

        for (label, hint) in [
            ("lang_guess", hints.lang_guess.as_deref()),
            ("Router(lang_guess=...)", self.lang_guess.as_deref()),
        ] {
            if let Some(english) = hint.and_then(english_from_code) {
                return Ok(RouteDecision {
                    model: if english {
                        RouteVariant::English
                    } else {
                        RouteVariant::Multilingual
                    },
                    reason: format!(
                        "{label}: the caller identified this as {} text",
                        if english { "English" } else { "non-English" }
                    ),
                    detection: None,
                    workflow: workflow.clone(),
                });
            }
        }

        let detection = language::analyse(state.as_content());
        let model = if detection.script == "unknown" {
            self.routing.default
        } else if detection.script != "latin" || !detection.is_english {
            RouteVariant::Multilingual
        } else if detection.language_undecided {
            self.routing.default
        } else {
            RouteVariant::English
        };
        Ok(RouteDecision {
            model,
            reason: detection_reason(&detection, self.routing.default),
            detection: Some(detection),
            workflow,
        })
    }

    fn path(&self, variant: RouteVariant) -> &PathBuf {
        match variant {
            RouteVariant::English => &self.english,
            RouteVariant::Multilingual => &self.multilingual,
            RouteVariant::TypedDecisions => &self.typed_decisions,
        }
    }

    fn load(&self, variant: RouteVariant) -> Result<Arc<LayaModel>, ModelError> {
        let mut loaded = self
            .loaded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(model) = loaded.get(&variant) {
            return Ok(model.clone());
        }

        let path = self.path(variant);
        tracing::info!(?variant, path = %path.display(), "loading Laya checkpoint");
        let model = Arc::new(LayaModel::load_dir(path, self.options.clone())?);
        loaded.insert(variant, model.clone());

        while loaded.len() > self.max_loaded {
            let Some(oldest) = loaded.keys().next().copied() else {
                break;
            };
            if oldest == variant {
                break;
            }
            loaded.shift_remove(&oldest);
        }

        Ok(model)
    }
}

impl DecisionModel for LayaRouter {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        let decision = self.route(state, questions)?;
        self.load(decision.model)?.predict(state, questions)
    }

    fn supports_cross_request_batching(&self) -> bool {
        !self.options.device.is_cpu()
    }

    fn predict_batch(
        &self,
        requests: &[Prediction<'_>],
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        let mut slots: Vec<Option<Result<DecisionOutput, ModelError>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut groups: HashMap<RouteVariant, Vec<usize>> = HashMap::new();
        for (index, request) in requests.iter().enumerate() {
            match self.route(request.state, request.questions) {
                Ok(decision) => groups.entry(decision.model).or_default().push(index),
                Err(error) => slots[index] = Some(Err(error)),
            }
        }

        for (variant, indices) in groups {
            let model = match self.load(variant) {
                Ok(model) => model,
                Err(error) => {
                    for index in indices {
                        slots[index] = Some(Err(error.clone()));
                    }
                    continue;
                }
            };
            let predictions: Vec<Prediction<'_>> = indices
                .iter()
                .map(|&index| Prediction::new(requests[index].state, requests[index].questions))
                .collect();
            for (index, output) in indices.into_iter().zip(model.predict_batch(&predictions)) {
                slots[index] = Some(output);
            }
        }

        slots
            .into_iter()
            .map(|slot| slot.expect("every request slot is filled"))
            .collect()
    }
}

fn match_typed_decisions_workflow(questions: &Questions) -> Option<String> {
    const WORKFLOWS: [(&str, &[&str]); 4] = [
        (
            "agent_trace_observability",
            &["action", "needs_review", "outcome", "risk", "urgency"],
        ),
        (
            "customer_service",
            &["action", "category", "churn_risk", "needs_human", "urgency"],
        ),
        (
            "invoice_processing",
            &[
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
        ),
        (
            "security_incidents",
            &[
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ),
    ];
    let ids: HashSet<&str> = questions.iter().map(|(id, _)| id.as_str()).collect();
    WORKFLOWS
        .iter()
        .find(|(_, signature)| {
            signature.len() == ids.len() && signature.iter().all(|id| ids.contains(id))
        })
        .map(|(name, _)| (*name).to_owned())
}

fn english_from_code(value: &str) -> Option<bool> {
    let code = value.trim().to_ascii_lowercase();
    if code.is_empty() {
        return None;
    }
    let code = code.split('.').next().unwrap_or("");
    let primary = code.replace('_', "-");
    let primary = primary.split('-').next().unwrap_or("");
    if primary.is_empty() || matches!(primary, "c" | "posix" | "und" | "zxx" | "mul") {
        return None;
    }
    Some(matches!(primary, "en" | "eng" | "english"))
}

fn python_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\x08"),
            '\u{0c}' => out.push_str("\\x0c"),
            '\u{0b}' => out.push_str("\\x0b"),
            '\u{07}' => out.push_str("\\x07"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch if !is_printable(ch) => push_unicode_escape(ch, &mut out),
            ch => out.push(ch),
        }
    }
    out.push(quote);
    out
}

fn is_printable(ch: char) -> bool {
    if ch == ' ' {
        return true;
    }
    !matches!(
        get_general_category(ch),
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

fn push_unicode_escape(ch: char, out: &mut String) {
    let cp = ch as u32;
    if cp <= 0xFF {
        out.push_str(&format!("\\x{cp:02x}"));
    } else if cp <= 0xFFFF {
        out.push_str(&format!("\\u{cp:04x}"));
    } else {
        out.push_str(&format!("\\U{cp:08x}"));
    }
}

fn detection_reason(analysis: &LanguageAnalysis, default: RouteVariant) -> String {
    if analysis.script == "unknown" {
        return format!(
            "no letters detected in state; using default ({})",
            default.as_str()
        );
    }
    if analysis.script != "latin" {
        return format!(
            "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
            analysis.script,
            non_latin_percent(analysis)
        );
    }
    if !analysis.is_english {
        if let Some(segment) = analysis.mixed_segment.as_deref() {
            let snippet: String = segment.chars().take(60).collect();
            return format!(
                "Latin script, mostly English, but a line or field reads as {} ({}); the English checkpoint cannot read it",
                python_repr(analysis.language.as_deref().unwrap_or("unknown")),
                python_repr(&snippet)
            );
        }
        if let Some(language) = analysis.language.as_deref() {
            return format!(
                "Latin script but language looks like {}, not English",
                python_repr(language)
            );
        }
        return format!(
            "Latin script, language not identified but {:.0}% non-English letters; not safe for the English checkpoint",
            100.0 * analysis.diacritic_rate
        );
    }
    if analysis.language_undecided {
        return format!(
            "Latin script, language not identified and no non-English letters; using default ({})",
            default.as_str()
        );
    }
    "English Latin text".to_owned()
}

fn non_latin_percent(analysis: &LanguageAnalysis) -> f64 {
    let latin = analysis.script_profile.get("latin").copied().unwrap_or(0.0);
    (1.0 - latin) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_detection_uses_function_words() {
        let english = language::analyse(&sys1_protocol::Content::String(
            "Please refund the duplicate charge on my account.".to_owned(),
        ));
        assert!(english.is_english);
        assert!(
            !language::analyse(&sys1_protocol::Content::String(
                "Bonjour, je voudrais un remboursement.".to_owned()
            ))
            .is_english
        );
        assert!(
            !language::analyse(&sys1_protocol::Content::String(
                "退款してください".to_owned()
            ))
            .is_english
        );
    }

    #[test]
    fn language_codes_abstain_when_they_name_no_language() {
        assert_eq!(english_from_code("en_US.UTF-8"), Some(true));
        assert_eq!(english_from_code("de-DE"), Some(false));
        assert_eq!(english_from_code("C.UTF-8"), None);
        assert_eq!(english_from_code("und"), None);
    }
}
