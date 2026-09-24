//! Which language a prompt is written in — as far as intent analysis needs it.
//!
//! Laya ships two checkpoints for routing: an English one and a multilingual one.
//! Its own `Router` picks between them by script, and on a Devanagari prompt it
//! picks correctly. It does not on the way Hindi is most often *typed* at a
//! plant: in Latin letters. Laya 0.3.20's `lang.py` carries stopword lists for
//! romanized Bangla, Azerbaijani and the Romance languages, but none for
//! romanized Hindi, so "is report ka summary do" comes back as "English Latin
//! text" — `is` is an English stopword — and goes to the English checkpoint.
//! Laya's own benchmark gives that checkpoint 0.100 on Hindi at 20 options,
//! against 0.050 for random guessing, and at high confidence.
//!
//! So ARJUN answers the one question routing needs — could this be Hindi? —
//! before Laya does, and names the multilingual checkpoint when it could.
//! Nothing here tries to be a general language identifier. English, Hindi in
//! Devanagari, and Hindi-English code-mixing are the three things a person at
//! this workbench writes; anything else is reported as `other` and left to
//! Laya's own detection.

use serde::{Deserialize, Serialize};

/// The language family of a prompt, as far as checkpoint choice cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromptLanguage {
    English,
    /// Mostly Devanagari.
    Hindi,
    /// Hindi and English mixed: romanized Hindi, or Devanagari and Latin words
    /// in one prompt.
    HindiEnglish,
    /// A non-Latin script that is not Devanagari. Laya's own detection handles it.
    Other,
}

impl PromptLanguage {
    /// BCP-47-style code for logs and the routing trace.
    pub fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Hindi => "hi",
            Self::HindiEnglish => "hi-en",
            Self::Other => "und",
        }
    }

    /// The Laya checkpoint ARJUN names for this language, or `None` to let
    /// Laya's own script detection choose.
    ///
    /// Only the Hindi cases are forced. Devanagari would reach the multilingual
    /// checkpoint anyway; naming it costs nothing and keeps the trace honest
    /// about who decided. Romanized Hindi would not, which is the point.
    pub fn forced_checkpoint(self) -> Option<&'static str> {
        match self {
            Self::Hindi | Self::HindiEnglish => Some("multilingual"),
            Self::English | Self::Other => None,
        }
    }
}

/// Romanized Hindi function words and common verbs.
///
/// Every word here is one an English sentence does not use. Left out on
/// purpose, because each is also ordinary English: `is` (this), `to` (then),
/// `me` (in), `the` (was), `main` (I), `do` (two / give), `par` (on), `hi`
/// (emphasis), `so`, `us`, `log` (people), `jab`, `bat`, `pal`, `band`, `din`.
/// A list that claimed any of them would move English prompts to the
/// multilingual checkpoint, which reads English measurably worse (0.657 vs
/// 0.783 on Laya's MASSIVE English intent benchmark).
const ROMAN_HINDI: &[&str] = &[
    "hai", "hain", "hoga", "hogi", "honge", "tha", "thi", "kya", "kyun", "kyon", "kaise",
    "kaisa", "kaisi", "kab", "kahan", "kaun", "kaunsa", "kaunsi", "kitna", "kitne", "kitni",
    "mujhe", "mera", "meri", "mere", "humein", "hamara", "hamare", "aap", "aapka", "aapki",
    "aapke", "tum", "tumhara", "tumhe", "yeh", "ye", "woh", "wo", "ka", "ki", "ke", "ko", "se",
    "mein", "nahi", "nahin", "aur", "bhi", "sirf", "karo", "karna", "karke", "kare", "karein",
    "karta", "karti", "karte", "kijiye", "kijie", "dijiye", "batao", "bataiye", "bataye",
    "likho", "likhna", "likhiye", "samjhao", "samjhaiye", "chahiye", "chahta", "chahti",
    "raha", "rahi", "rahe", "gaya", "gayi", "diya", "liya", "wala", "wali", "wale", "abhi",
    "phir", "jaldi", "accha", "acha", "theek", "thik", "ek", "saath", "liye", "agar", "toh",
    "lekin", "yaar", "bhai", "haan", "nikalo", "dekho", "banao", "chalao", "sahi", "galat",
    "kuch", "zyada", "hisaab", "hisab", "dono", "kaam", "ho", "kar", "tak", "wahi", "yahi",
];

/// Share of letters that must be Devanagari for a prompt to count as Hindi
/// rather than mixed. Below it and above zero, the prompt mixes scripts.
const DEVANAGARI_MAJORITY: f32 = 0.6;

/// Classifies a prompt's language for checkpoint choice.
///
/// Deterministic, allocation-light and dependency-free: it runs on every turn
/// before routing, so it has to cost microseconds.
pub fn detect(prompt: &str) -> PromptLanguage {
    let mut latin = 0usize;
    let mut devanagari = 0usize;
    let mut other = 0usize;
    for ch in prompt.chars().filter(|c| c.is_alphabetic()) {
        match ch as u32 {
            0x0900..=0x097F | 0xA8E0..=0xA8FF => devanagari += 1,
            c if c < 0x0250 || (0x1E00..=0x1EFF).contains(&c) => latin += 1,
            _ => other += 1,
        }
    }
    let letters = latin + devanagari + other;
    if letters == 0 {
        return PromptLanguage::English;
    }
    if devanagari > 0 {
        let share = devanagari as f32 / letters as f32;
        return if share >= DEVANAGARI_MAJORITY {
            PromptLanguage::Hindi
        } else {
            PromptLanguage::HindiEnglish
        };
    }
    if other > latin {
        return PromptLanguage::Other;
    }

    // Latin script: count distinct romanized-Hindi markers among the words.
    let lowered = prompt.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return PromptLanguage::English;
    }
    let mut seen: Vec<&str> = Vec::new();
    for word in &words {
        if ROMAN_HINDI.contains(word) && !seen.contains(word) {
            seen.push(word);
        }
    }
    // Two distinct markers is Hindi sentence structure ("kya hai", "karo ...
    // ke liye"). One is enough when it is at least a fifth of the words: "is
    // report ka summary do" has only `ka` once `is` and `do` are excluded as
    // English, and it is the commonest shape of a Hinglish request. One marker
    // in a long English sentence ("Tell me about Ek Onkar and its history") is
    // more likely a name than grammar, and stays English.
    let markers = seen.len();
    if markers >= 2 || (markers == 1 && markers * 5 >= words.len()) {
        PromptLanguage::HindiEnglish
    } else {
        PromptLanguage::English
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devanagari_is_hindi() {
        assert_eq!(detect("इस निरीक्षण रिपोर्ट का सारांश दो"), PromptLanguage::Hindi);
        assert_eq!(detect("नमस्ते"), PromptLanguage::Hindi);
    }

    #[test]
    fn devanagari_with_english_terms_is_mixed() {
        assert_eq!(
            detect("इस Python script में error आ रहा है, fix करो"),
            PromptLanguage::HindiEnglish
        );
    }

    /// The case Laya's own router sends to the English checkpoint.
    #[test]
    fn romanized_hindi_is_mixed_not_english() {
        for prompt in [
            "is report ka summary do",
            "mujhe ek python function likhna hai jo list sort kare",
            "x ka value nikalo agar 3x + 5 = 20 hai",
            "pressure vessel ki inspection method kya hai",
            "batao",
        ] {
            assert_eq!(detect(prompt), PromptLanguage::HindiEnglish, "{prompt:?}");
        }
    }

    /// The other half: English that happens to contain a listed-out collision
    /// word stays English, so it keeps the checkpoint that reads English best.
    #[test]
    fn english_stays_english() {
        for prompt in [
            "Write a Python function to parse this CSV",
            "What is the main reason this is failing? Do the rest too.",
            "Summarise the inspection report for the crude distillation unit",
            "Tell me about Ek Onkar and its history in Punjab",
            "ok",
            "",
            "3x + 5 = 20",
        ] {
            assert_eq!(detect(prompt), PromptLanguage::English, "{prompt:?}");
        }
    }

    #[test]
    fn other_scripts_are_left_to_laya() {
        assert_eq!(detect("Как дела?"), PromptLanguage::Other);
        assert_eq!(PromptLanguage::Other.forced_checkpoint(), None);
    }

    #[test]
    fn only_hindi_forces_a_checkpoint() {
        assert_eq!(PromptLanguage::Hindi.forced_checkpoint(), Some("multilingual"));
        assert_eq!(PromptLanguage::HindiEnglish.forced_checkpoint(), Some("multilingual"));
        assert_eq!(PromptLanguage::English.forced_checkpoint(), None);
    }
}
