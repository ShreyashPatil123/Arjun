//! The HuggingFace token, resolved once for the whole process.
//!
//! Reaching the Hub happens from several places — browsing the catalog,
//! discovering adapters, resolving artifacts, downloading weights — and most of
//! them are plain commands with no access to the Tauri app handle needed to
//! read `config.json`. Rather than thread a handle through all of them, the
//! saved token is loaded into memory at startup and read from here.
//!
//! Without a token the Hub's anonymous rate limit caps a browse sweep, so the
//! catalog can only show a small popular slice of the library.

use std::sync::{OnceLock, RwLock};

/// Variable names the official `huggingface_hub` tooling uses, so an existing
/// shell setup is picked up without configuring anything.
const ENV_KEYS: [&str; 3] = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"];

fn slot() -> &'static RwLock<Option<String>> {
    static SLOT: OnceLock<RwLock<Option<String>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

fn clean(token: Option<String>) -> Option<String> {
    token.map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// Replaces the saved token. `None` clears it, falling back to the environment.
pub fn set(token: Option<String>) {
    if let Ok(mut guard) = slot().write() {
        *guard = clean(token);
    }
}

/// The token to authenticate Hub requests with, if there is one.
///
/// The setting wins over the environment: someone who typed a token into
/// Settings expects that one to be used, even on a machine where a stale
/// `HF_TOKEN` is exported.
pub fn get() -> Option<String> {
    if let Ok(guard) = slot().read() {
        if let Some(token) = guard.as_ref() {
            return Some(token.clone());
        }
    }
    clean(ENV_KEYS.iter().find_map(|k| std::env::var(k).ok()))
}

/// Whether a token is available from any source.
pub fn is_present() -> bool {
    get().is_some()
}

/// Reports where the active token came from, for display in Settings.
///
/// Distinguishing these matters: a user who cleared the field but still has a
/// token exported would otherwise see "not configured" while requests continue
/// to be authenticated.
pub fn source() -> &'static str {
    let from_setting = slot().read().ok().map(|g| g.is_some()).unwrap_or(false);
    if from_setting {
        "settings"
    } else if clean(ENV_KEYS.iter().find_map(|k| std::env::var(k).ok())).is_some() {
        "environment"
    } else {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests share one process-wide slot, so they run under a lock and
    /// restore it rather than leaving state behind for whatever runs next.
    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_saved_token_is_returned() {
        let _lock = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        set(Some("hf_example".to_string()));
        assert_eq!(get().as_deref(), Some("hf_example"));
        set(None);
    }

    #[test]
    fn surrounding_whitespace_is_trimmed_away() {
        // Pasting from a browser commonly brings a trailing newline, which the
        // Hub rejects as a malformed Authorization header.
        let _lock = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        set(Some("  hf_example\n".to_string()));
        assert_eq!(get().as_deref(), Some("hf_example"));
        set(None);
    }

    #[test]
    fn an_empty_setting_is_the_same_as_no_setting() {
        let _lock = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        set(Some("   ".to_string()));
        assert!(slot().read().unwrap().is_none(), "blank input must not count as a token");
        set(None);
    }
}
