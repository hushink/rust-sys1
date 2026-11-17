//! Dependency-free script/Latin-language detection for the Laya router.
//!
//! The heuristic and its stopword tables reproduce the Apache-2.0 Laya router's
//! routing behaviour so the routing golden vectors can be replayed offline.

use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use sys1_protocol::Content;
use unicode_general_category::{GeneralCategory, get_general_category};

fn is_letter(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

const NON_EN_DIACRITIC_RATE: f64 = 0.02;
const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;
const NON_LATIN_FRACTION: f64 = 0.2;
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
const NON_LATIN_MIN_LETTERS: usize = 10;
const SHOUTED_MIN_STOPWORD: usize = 4;
const SHOUTED_MIN_WORD: usize = 5;

const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

const EN_STOP: &[&str] = &[
    "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that", "this",
    "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from", "will", "can",
    "would", "there", "their", "what", "which", "please", "we", "i", "how", "why", "when", "where",
    "who", "whom", "does", "done", "doing", "been", "being", "could", "should", "might", "must",
    "shall", "doesn", "didn", "isn", "aren", "wasn", "weren", "haven", "hasn", "couldn", "wouldn",
    "won", "dont", "wont", "didnt", "doesnt", "isnt", "wasnt", "havent", "ive", "youre", "thats",
    "whats", "any", "some", "many", "much", "every", "each", "these", "those", "both", "few",
    "they", "them", "she", "him", "his", "our", "your", "yours", "into", "onto", "between",
    "after", "before", "through", "during", "without", "within", "down", "if", "nor", "than",
    "then", "because", "while", "until", "here", "only", "very", "too", "ever", "never", "always",
    "already", "yet",
];

const FR_STOP: &[&str] = &[
    "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur", "pas",
    "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce", "et", "du", "au",
    "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton", "ma", "ta", "sa", "mes", "tes",
    "ses", "ces", "deux", "trois", "très", "bien", "tout", "tous", "toute", "fait", "veux", "veut",
    "peux", "peut", "dois", "doit", "merci", "bonjour", "jour", "jours", "mois", "fois", "quand",
    "comment", "pourquoi", "alors", "donc",
];

const DE_STOP: &[&str] = &[
    "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für", "auf",
    "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber", "ich", "wir",
    "mir", "mich", "dir", "dich", "uns", "mein", "meine", "meinen", "meinem", "meiner", "diese",
    "dieser", "diesen", "dieses", "einen", "einem", "einer", "wie", "wo", "wann", "welche", "im",
    "zum", "zur", "aus", "bei", "nach", "noch", "bitte", "heute", "jetzt", "kann", "kannst",
    "habe", "gibt", "wird", "in", "was",
];

const ES_STOP: &[&str] = &[
    "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como", "pero",
    "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus", "la", "un", "y", "al", "lo",
    "le", "les", "su", "mi", "tu", "nos", "ni", "dos", "tres", "fue", "fueron", "ser", "tiene",
    "tienen", "tengo", "puede", "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre",
    "cuando", "donde", "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada",
    "algo", "aquí", "hoy", "gracias",
];

const PT_STOP: &[&str] = &[
    "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da", "dos",
    "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela", "o", "e", "na", "nas",
    "nos", "ao", "aos", "por", "foi", "era", "ser", "sou", "tem", "tenho", "pode", "podem",
    "quero", "preciso", "eu", "meu", "minha", "seu", "sua", "isso", "isto", "aqui", "ali", "como",
    "quando", "onde", "porque", "mais", "já", "ainda", "agora", "hoje", "ontem", "dois", "três",
    "tudo", "nada", "obrigado", "olá", "você", "vocês", "voce", "voces", "vc", "vcs", "nao", "sao",
    "ja", "até", "tá", "pra", "gostaria", "obrigada", "também", "tambem", "estou", "estamos",
    "meus", "minhas", "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite", "depois",
    "antes", "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum", "nenhuma",
    "estava", "ficou", "fiz", "deu",
];

const IT_STOP: &[&str] = &[
    "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
    "questo", "questa", "anche", "come", "più", "nella", "alla", "la", "le", "un", "uno", "una",
    "e", "ed", "o", "da", "su", "tra", "fra", "mi", "ci", "ne", "ho", "hai", "ha", "abbiamo",
    "avete", "hanno", "era", "stato", "stata", "devo", "deve", "devono", "voglio", "vorrei", "mio",
    "mia", "tuo", "sua", "quando", "dove", "perche", "molto", "poco", "sempre", "mai", "già",
    "ancora", "adesso", "oggi", "ieri", "grazie", "ciao", "scusa", "nel", "nell", "negli", "sul",
    "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei", "delle", "dello", "degli", "agli",
    "alle", "col",
];

const NL_STOP: &[&str] = &[
    "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan", "door",
    "maar", "ook", "worden", "deze", "naar", "wordt",
];

const SV_STOP: &[&str] = &[
    "jag",
    "är",
    "och",
    "inte",
    "att",
    "från",
    "till",
    "behöver",
    "får",
    "skulle",
    "ska",
    "vill",
    "måste",
    "också",
    "dessa",
    "detta",
    "säger",
    "upp",
    "utan",
    "mitt",
    "min",
    "om",
    "kommer",
    "här",
    "två",
    "vi",
    "nästa",
    "gör",
    "göra",
    "hjälp",
    "hjälpa",
    "mig",
    "återbetalning",
    "återbetala",
    "faktura",
    "gång",
    "gånger",
    "hittar",
    "inställningen",
    "inställningarna",
    "lösenord",
    "när",
    "öppnar",
    "spårningen",
    "aterbetalning",
    "aterbetala",
    "behover",
    "fel",
    "ganger",
    "hjalp",
    "hjalpa",
    "installningen",
    "installningarna",
    "kraschar",
    "kvittot",
    "losenord",
    "nar",
    "oppnar",
    "paket",
    "skicka",
    "sparningen",
    "tva",
    "uppdaterats",
    "blivit",
    "debiterade",
    "appen",
];

const RO_STOP: &[&str] = &[
    "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără", "ale",
    "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această", "acesta",
    "aceasta", "mi", "ți", "vă", "nu",
];

const BN_STOP: &[&str] = &[
    "ami",
    "amar",
    "amake",
    "amra",
    "amader",
    "apni",
    "apnar",
    "apnake",
    "apnara",
    "tumi",
    "tomar",
    "tomake",
    "tomra",
    "tader",
    "ota",
    "eita",
    "oita",
    "ekta",
    "ei",
    "oi",
    "ki",
    "keno",
    "kivabe",
    "kibhabe",
    "kothay",
    "kokhon",
    "kobe",
    "koto",
    "kintu",
    "jodi",
    "tahole",
    "ar",
    "theke",
    "jonno",
    "sathe",
    "shathe",
    "diye",
    "niye",
    "moddhe",
    "kore",
    "korte",
    "korchi",
    "korsi",
    "korbo",
    "korechi",
    "koreche",
    "korun",
    "koren",
    "korlam",
    "hobe",
    "hoyeche",
    "hoise",
    "hocche",
    "hoyni",
    "chai",
    "chaina",
    "lagbe",
    "parchi",
    "parbo",
    "parchina",
    "peyechi",
    "paini",
    "dite",
    "dilam",
    "diyechi",
    "nai",
    "khub",
    "onek",
    "ekhon",
    "akhon",
    "ekhono",
    "abar",
    "ekbar",
    "duibar",
    "ajke",
    "kalke",
    "taka",
    "bhalo",
    "valo",
    "kharap",
    "shomossa",
    "somossa",
    "dhonnobad",
    "bhai",
    "shob",
    "keu",
    "kichu",
    "bolte",
    "bolun",
    "parben",
    "asbe",
    "jabe",
    "pabo",
    "ferot",
    "dorkar",
    "hoye",
    "geche",
    "gese",
];

const AZ_STOP: &[&str] = &[
    "və", "ve", "bir", "bu", "üçün", "ucun", "ilə", "ile", "olan", "olub", "olmasa", "var", "yox",
    "yoxdur", "mən", "sən", "biz", "siz", "onlar", "daha", "çox", "cox", "hər", "nə", "kimi",
    "görə", "sonra", "əgər", "eger", "deyil", "lakin", "amma", "ancaq", "artıq", "artiq", "də",
    "isə", "həm", "yalnız", "yalniz",
];

const SHORT_SWEDISH_WORDS: &[&str] = &[
    "åtkomst",
    "atkomst",
    "lösenord",
    "losenord",
    "fakturan",
    "betalningen",
    "inloggningen",
    "glömt",
    "glomt",
    "behöver",
    "behover",
    "återbetalning",
    "aterbetalning",
    "kvitto",
    "kvittot",
    "spårningen",
    "sparningen",
    "inställningen",
    "installningen",
    "felmeddelande",
    "abonnemanget",
];

const NORDIC_OVERLAP_WORDS: &[&str] = &[
    "hej", "ja", "nej", "jo", "tack", "mig", "min", "om", "kommer", "får", "skulle", "vi",
];

const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ\
ăâîșțşţąćęłńśźżčďěňřšťůžőűğıāēģīķļņūžđə\
ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹơư";

fn stop_sets() -> &'static HashMap<&'static str, HashSet<&'static str>> {
    static STOP: OnceLock<HashMap<&'static str, HashSet<&'static str>>> = OnceLock::new();
    STOP.get_or_init(|| {
        let mut map = HashMap::new();
        for (name, words) in [
            ("en", EN_STOP),
            ("fr", FR_STOP),
            ("de", DE_STOP),
            ("es", ES_STOP),
            ("pt", PT_STOP),
            ("it", IT_STOP),
            ("nl", NL_STOP),
            ("sv", SV_STOP),
            ("ro", RO_STOP),
            ("bn", BN_STOP),
            ("az", AZ_STOP),
        ] {
            map.insert(name, words.iter().copied().collect());
        }
        map
    })
}

fn shared_words() -> &'static HashSet<&'static str> {
    static SHARED: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SHARED.get_or_init(|| {
        let stop = stop_sets();
        let mut shared = HashSet::new();
        for words in stop.values() {
            for word in words {
                if stop.values().filter(|other| other.contains(word)).count() > 1 {
                    shared.insert(*word);
                }
            }
        }
        shared.extend(NORDIC_OVERLAP_WORDS.iter().copied());
        shared
    })
}

fn en_only_words() -> &'static HashSet<&'static str> {
    static EN_ONLY: OnceLock<HashSet<&'static str>> = OnceLock::new();
    EN_ONLY.get_or_init(|| {
        let shared = shared_words();
        EN_STOP
            .iter()
            .copied()
            .filter(|word| !shared.contains(word))
            .collect()
    })
}

fn en_collision_words() -> &'static HashSet<&'static str> {
    static COLLISIONS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    COLLISIONS.get_or_init(|| {
        let stop = stop_sets();
        ["come", "son", "do", "care", "todo", "im", "per", "plus"]
            .into_iter()
            .filter(|word| {
                stop.iter()
                    .any(|(lang, words)| *lang != "en" && words.contains(word))
            })
            .collect()
    })
}

/// Public detection result. Mirrors the Python router's `analyse` fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LanguageAnalysis {
    pub script: String,
    pub script_profile: IndexMap<String, f64>,
    pub language: Option<String>,
    pub is_english: bool,
    pub language_undecided: bool,
    pub diacritic_rate: f64,
    pub non_latin_fraction: f64,
    pub mixed_segment: Option<String>,
}

#[derive(Default)]
struct ScriptCounts {
    counts: IndexMap<String, usize>,
    latin: usize,
}

impl ScriptCounts {
    fn add(&mut self, script: &str) {
        *self.counts.entry(script.to_owned()).or_insert(0) += 1;
    }

    fn finish(mut self) -> Self {
        self.counts.insert("latin".to_owned(), self.latin);
        self
    }

    fn total(&self) -> usize {
        self.counts.values().sum()
    }

    fn dominant(&self) -> &str {
        if self.total() == 0 {
            return "unknown";
        }
        let mut best = ("unknown", 0);
        for (name, count) in &self.counts {
            if *count > best.1 {
                best = (name.as_str(), *count);
            }
        }
        best.0
    }

    fn profile(&self) -> IndexMap<String, f64> {
        let total = self.total();
        if total == 0 {
            return IndexMap::new();
        }
        let mut profile = IndexMap::new();
        if let Some(latin) = self.counts.get("latin")
            && *latin > 0
        {
            profile.insert("latin".to_owned(), *latin as f64 / total as f64);
        }
        for (name, count) in &self.counts {
            if name != "latin" && *count > 0 {
                profile.insert(name.clone(), *count as f64 / total as f64);
            }
        }
        profile
    }
}

fn range_script(cp: u32) -> Option<&'static str> {
    for (script, ranges) in SCRIPT_RANGES {
        if ranges.iter().any(|(lo, hi)| (*lo..=*hi).contains(&cp)) {
            return Some(script);
        }
    }
    None
}

fn script_counts(text: &str) -> ScriptCounts {
    let mut counts = ScriptCounts::default();
    for ch in text.chars() {
        if !is_letter(ch) {
            continue;
        }
        let cp = ch as u32;
        if cp < 0x02B0
            || (0x1E00..=0x1EFF).contains(&cp)
            || (0xFF21..=0xFF3A).contains(&cp)
            || (0xFF41..=0xFF5A).contains(&cp)
        {
            counts.latin += 1;
            continue;
        }
        match range_script(cp) {
            Some(script) => counts.add(script),
            None => counts.add("other"),
        }
    }
    counts.finish()
}

fn script_of(ch: char) -> Option<&'static str> {
    let cp = ch as u32;
    if cp < 0x0250
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
    {
        return None;
    }
    range_script(cp)
}

fn leaf_value(value: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match value {
        Value::String(text) => out.push(text.clone()),
        Value::Array(values) => {
            for value in values {
                leaf_value(value, depth + 1, out);
            }
        }
        Value::Object(map) => {
            for value in map.values() {
                leaf_value(value, depth + 1, out);
            }
        }
        _ => {}
    }
}

fn leaves(content: &Content) -> Vec<String> {
    let mut out = Vec::new();
    match content {
        Content::String(text) => out.push(text.clone()),
        Content::Array(values) => {
            for value in values {
                leaf_value(value, 1, &mut out);
            }
        }
        Content::Object(map) => {
            for value in map.values() {
                leaf_value(value, 1, &mut out);
            }
        }
        Content::Null => {}
    }
    out
}

fn state_text(content: &Content, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if let Content::String(text) = content {
        return text.chars().take(max_chars).collect();
    }
    let mut parts = Vec::new();
    let mut budget = max_chars;
    for leaf in leaves(content) {
        if budget == 0 {
            break;
        }
        let chars: Vec<char> = leaf.chars().collect();
        if chars.len() > budget {
            parts.push(chars[..budget].iter().collect::<String>());
            break;
        }
        parts.push(leaf);
        budget = budget.saturating_sub(chars.len() + 1);
    }
    let joined = parts.join(" ");
    joined.chars().take(max_chars).collect()
}

fn alpha_count(text: &str) -> usize {
    text.chars().filter(|ch| is_letter(*ch)).count()
}

fn round4(value: f64) -> f64 {
    format!("{value:.4}").parse().unwrap_or(value)
}

fn script_profile(text: &str) -> (ScriptCounts, IndexMap<String, f64>, &'static str) {
    let counts = script_counts(text);
    let profile = counts.profile();
    let dominant = counts.dominant();
    let script = if dominant == "unknown" {
        "unknown"
    } else if dominant == "latin" {
        "latin"
    } else {
        SCRIPT_RANGES
            .iter()
            .map(|(script, _)| *script)
            .find(|script| *script == dominant)
            .unwrap_or("other")
    };
    (counts, profile, script)
}

fn non_latin_carries(non_latin: f64, n_non_latin: usize) -> bool {
    non_latin >= NON_LATIN_FRACTION
        || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS)
}

fn non_latin_words(text: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    let mut script: Option<&str> = None;
    for ch in text.chars() {
        let next = script_of(ch);
        if next.is_some() && next == script {
            current.push(ch);
            continue;
        }
        if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
        if let Some(next) = next {
            current.push(ch);
            script = Some(next);
        } else {
            script = None;
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs.into_iter()
        .filter(|word| {
            word.chars().count() >= 2 && !word.chars().next().is_some_and(|ch| ch.is_uppercase())
        })
        .collect()
}

fn is_identifier_token(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if (*ch == '.' || *ch == '@') && index > 0 && index + 1 < chars.len() {
            let before = chars[index - 1];
            let after = chars[index + 1];
            if (before.is_alphanumeric() || before == '-' || before == '_')
                && (after.is_alphanumeric() || after == '-' || after == '_')
            {
                return true;
            }
        }
    }
    false
}

fn words(text: &str) -> Vec<String> {
    let replaced = text.replace('İ', "i");
    let cleaned = replaced
        .split_whitespace()
        .filter(|token| !is_identifier_token(token))
        .collect::<Vec<_>>()
        .join(" ");
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in cleaned.chars() {
        if is_letter(ch) {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn diacritic_rate(text: &str) -> f64 {
    let lowered: Vec<char> = text.chars().flat_map(char::to_lowercase).collect();
    let diacritics = NON_EN_DIACRITICS.chars().collect::<HashSet<_>>();
    let count = lowered.iter().filter(|ch| diacritics.contains(ch)).count();
    count as f64 / lowered.len().max(1) as f64
}

struct LatinProfile {
    language: Option<String>,
    diacritic_rate: f64,
    looks_non_english: bool,
}

fn latin_profile(text: &str) -> LatinProfile {
    let words = words(text);
    let rate = diacritic_rate(text);
    let non_english = rate >= NON_EN_DIACRITIC_RATE;
    let en_only = en_only_words();
    let nordic_overlap = words
        .iter()
        .any(|word| NORDIC_OVERLAP_WORDS.contains(&word.as_str()))
        && !words.iter().any(|word| en_only.contains(word.as_str()));
    if (2..4).contains(&words.len())
        && words
            .iter()
            .any(|word| SHORT_SWEDISH_WORDS.contains(&word.as_str()))
    {
        return LatinProfile {
            language: Some("sv".to_owned()),
            diacritic_rate: rate,
            looks_non_english: non_english,
        };
    }
    if words.len() < 4 {
        return LatinProfile {
            language: None,
            diacritic_rate: rate,
            looks_non_english: non_english || nordic_overlap,
        };
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for word in &words {
        *counts.entry(word.as_str()).or_insert(0) += 1;
    }
    let collisions = en_collision_words();
    let stop = stop_sets();
    const LANGUAGES: [&str; 11] = [
        "en", "fr", "de", "es", "pt", "it", "nl", "sv", "ro", "bn", "az",
    ];
    let mut scores = Vec::with_capacity(LANGUAGES.len());
    for lang in LANGUAGES {
        let table = &stop[lang];
        let mut score = 0;
        for (word, count) in &counts {
            if table.contains(word) {
                score += if collisions.contains(word) { 1 } else { *count };
            }
        }
        scores.push((lang, score));
    }
    let en = scores
        .iter()
        .find(|(lang, _)| *lang == "en")
        .map(|(_, score)| *score)
        .unwrap_or(0);
    let shared = shared_words();
    let evidenced: Vec<(&str, usize)> = scores
        .iter()
        .filter(|(lang, _)| *lang != "en")
        .filter(|(lang, _)| {
            words.iter().any(|word| {
                stop.get(*lang)
                    .is_some_and(|table| table.contains(word.as_str()))
                    && !shared.contains(word.as_str())
            })
        })
        .copied()
        .collect();
    let mut best: Option<(&str, usize)> = None;
    for candidate in &evidenced {
        if best.is_none_or(|current| candidate.1 > current.1) {
            best = Some(*candidate);
        }
    }

    let mut language = None;
    if let Some((lang, score)) = best {
        let swedish_short = lang == "sv"
            && words.iter().any(|word| word == "inte")
            && words.iter().any(|word| word == "kan")
            && matches!(
                words.first().map(String::as_str),
                Some("kan" | "jag" | "vi")
            )
            && en <= 1;
        if score >= (en + 2).max(2) || swedish_short || (non_english && score >= en.max(2)) {
            language = Some(lang.to_owned());
        }
    }
    if language.is_none() && en > 0 && (!non_english || english_rescued_by_words(&words, rate)) {
        language = Some("en".to_owned());
    }
    let undecided = language.is_none();
    LatinProfile {
        language,
        diacritic_rate: rate,
        looks_non_english: non_english || (undecided && nordic_overlap),
    }
}

fn english_rescued_by_words(words: &[String], rate: f64) -> bool {
    if rate >= ENGLISH_RESCUE_DIACRITIC_RATE {
        return false;
    }
    let en_only = en_only_words();
    let unique: HashSet<&str> = words.iter().map(String::as_str).collect();
    if unique
        .iter()
        .filter(|word| en_only.contains(**word))
        .count()
        < 2
    {
        return false;
    }
    let diacritics: HashSet<char> = NON_EN_DIACRITICS.chars().collect();
    unique
        .iter()
        .filter(|word| word.chars().any(|ch| diacritics.contains(&ch)))
        .count()
        <= 1
}

fn shouted(text: &str) -> bool {
    text.chars().any(|ch| ch.is_uppercase()) && !text.chars().any(|ch| ch.is_lowercase())
}

fn shouted_evidence(tokens: &[String], lang: &str, rate: f64) -> bool {
    if !tokens
        .iter()
        .any(|token| token.chars().count() >= SHOUTED_MIN_WORD)
    {
        return false;
    }
    if rate >= NON_EN_DIACRITIC_RATE {
        return true;
    }
    let Some(table) = stop_sets().get(lang) else {
        return false;
    };
    tokens
        .iter()
        .map(|token| token.to_lowercase())
        .any(|word| table.contains(word.as_str()) && word.chars().count() >= SHOUTED_MIN_STOPWORD)
}

fn code_line(text: &str) -> bool {
    if text
        .chars()
        .any(|ch| matches!(ch, '=' | ';' | '{' | '}' | '[' | ']'))
    {
        return true;
    }
    let chars: Vec<char> = text.chars().collect();
    chars
        .windows(2)
        .any(|pair| (pair[0].is_alphanumeric() || pair[0] == '_') && pair[1] == '(')
}

fn joined_token(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    chars.windows(3).any(|window| {
        (window[0].is_alphanumeric() || window[0] == '_')
            && matches!(window[1], '.' | '_' | '/' | '\\')
            && (window[2].is_alphanumeric() || window[2] == '_')
    })
}

fn letter_runs(text: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if is_letter(ch) {
            current.push(ch);
        } else if !current.is_empty() {
            if current.chars().count() >= 2 {
                runs.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.chars().count() >= 2 {
        runs.push(current);
    }
    runs
}

fn blank_acronyms(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut current = String::new();
    for ch in text.chars() {
        if is_letter(ch) {
            current.push(ch);
        } else {
            if current.chars().count() >= 2
                && current.chars().all(|ch| ch.is_uppercase())
                && current.chars().count() < SHOUTED_MIN_WORD
            {
                out.push(' ');
            } else {
                out.push_str(&current);
            }
            current.clear();
            out.push(ch);
        }
    }
    if current.chars().count() >= 2
        && current.chars().all(|ch| ch.is_uppercase())
        && current.chars().count() < SHOUTED_MIN_WORD
    {
        out.push(' ');
    } else {
        out.push_str(&current);
    }
    out
}

fn named_prose_language(segment: &str) -> Option<String> {
    if segment.trim().is_empty() || code_line(segment) {
        return None;
    }
    let prose = segment
        .split_whitespace()
        .filter(|token| !joined_token(token))
        .collect::<Vec<_>>()
        .join(" ");
    let is_shouted = shouted(&prose);
    let prose = if is_shouted {
        prose
    } else {
        blank_acronyms(&prose)
    };
    let tokens = words(&prose);
    if tokens.len() < 4 {
        return None;
    }
    let profile = latin_profile(&prose);
    let lang = profile.language?;
    if lang == "en" {
        return None;
    }
    if is_shouted && !shouted_evidence(&tokens, &lang, profile.diacritic_rate) {
        return None;
    }
    let table = stop_sets().get(lang.as_str())?;
    let unique: HashSet<String> = tokens.iter().map(|token| token.to_lowercase()).collect();
    if unique
        .iter()
        .filter(|word| table.contains(word.as_str()))
        .count()
        < 2
    {
        return None;
    }
    Some(lang)
}

fn non_english_segment(content: &Content) -> Option<(String, String)> {
    let mut seen = 0usize;
    for leaf in leaves(content) {
        for segment in leaf.split('\n') {
            if seen >= 4000 {
                return None;
            }
            let segment: String = segment.chars().take(4000 - seen).collect();
            seen += segment.chars().count();
            if let Some(lang) = named_prose_language(&segment) {
                return Some((lang, segment.trim().to_owned()));
            }
        }
    }
    None
}

fn analyse_text(text: &str) -> LanguageAnalysis {
    let (_counts, profile, mut script) = script_profile(text);
    let non_latin = if profile.is_empty() {
        0.0
    } else {
        round4(1.0 - profile.get("latin").copied().unwrap_or(0.0))
    };
    let n_non_latin = (non_latin * alpha_count(text) as f64).round() as usize;
    if script == "latin"
        && !non_latin_words(text).is_empty()
        && non_latin_carries(non_latin, n_non_latin)
    {
        let mut best = ("other", 0.0);
        for (name, value) in &profile {
            if name != "latin" && *value > best.1 {
                best = (name.as_str(), *value);
            }
        }
        script = best.0;
    }
    if script == "unknown" {
        return LanguageAnalysis {
            script: "unknown".to_owned(),
            script_profile: profile,
            language: None,
            is_english: true,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: 0.0,
            mixed_segment: None,
        };
    }
    if script != "latin" {
        return LanguageAnalysis {
            script: script.to_owned(),
            script_profile: profile,
            language: None,
            is_english: false,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: non_latin,
            mixed_segment: None,
        };
    }
    let profile_lat = latin_profile(text);
    let mut language = profile_lat.language.clone();
    if let Some(lang) = language.as_deref()
        && lang != "en"
        && !non_latin_carries(non_latin, n_non_latin)
    {
        if shouted(text) {
            if !shouted_evidence(&words(text), lang, profile_lat.diacritic_rate) {
                language = None;
            }
        } else {
            let caps: Vec<String> = letter_runs(text)
                .into_iter()
                .filter(|run| run.chars().all(|ch| ch.is_uppercase()))
                .collect();
            if !caps.is_empty()
                && !caps
                    .iter()
                    .any(|run| run.chars().count() >= SHOUTED_MIN_WORD)
            {
                let blanked = blank_acronyms(text);
                if blanked != text {
                    language = latin_profile(&blanked).language;
                }
            }
        }
    }
    let undecided = language.is_none();
    let english =
        language.as_deref() == Some("en") || (undecided && !profile_lat.looks_non_english);
    LanguageAnalysis {
        script: "latin".to_owned(),
        script_profile: profile,
        language,
        is_english: english,
        language_undecided: undecided,
        diacritic_rate: round4(profile_lat.diacritic_rate),
        non_latin_fraction: non_latin,
        mixed_segment: None,
    }
}

fn leaf_non_english(leaf: &str) -> Option<LanguageAnalysis> {
    let mut best_n = 0usize;
    let mut best = None;
    for line in leaf.split('\n') {
        if line.chars().count() < 7 {
            continue;
        }
        let sample: String = line.chars().take(4000).collect();
        if sample.trim().is_empty() || code_line(&sample) {
            continue;
        }
        let det = analyse_text(&sample);
        if det.is_english {
            continue;
        }
        let named = det.language.as_deref() != Some("en")
            && det.language.is_some()
            && named_prose_language(&sample).is_some();
        if !named {
            if det.script != "latin" && det.script != "unknown" {
                if !(!non_latin_words(&sample).is_empty()
                    && alpha_count(&sample) >= NON_LATIN_MIN_LETTERS)
                {
                    continue;
                }
            } else if !(det.diacritic_rate >= NON_EN_DIACRITIC_RATE && words(&sample).len() >= 4) {
                continue;
            }
        }
        let n_alpha = alpha_count(&sample);
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    best
}

/// Full analysis of a state value.
pub(crate) fn analyse(content: &Content) -> LanguageAnalysis {
    let mut result = analyse_text(&state_text(content, 4000));
    if result.script == "latin" && result.is_english {
        let leaves = leaves(content);
        let structured = leaves.len() > 1 || leaves.first().is_some_and(|leaf| leaf.contains('\n'));
        if structured && let Some((language, mixed)) = non_english_segment(content) {
            result.language = Some(language);
            result.is_english = false;
            result.language_undecided = false;
            result.mixed_segment = Some(mixed);
        }
    }
    if matches!(content, Content::String(_)) || content == &Content::Null || !result.is_english {
        return result;
    }
    let mut best_n = 0usize;
    let mut best = None;
    for leaf in leaves(content) {
        let Some(det) = leaf_non_english(&leaf) else {
            continue;
        };
        let n_alpha = alpha_count(&leaf.chars().take(4000).collect::<String>());
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    if let Some(best) = best {
        result.language = best.language;
        result.is_english = false;
        result.language_undecided = best.language_undecided;
    }
    result
}
