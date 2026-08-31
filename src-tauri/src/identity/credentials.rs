//! Proving that somebody is who they say they are.
//!
//! [`super`] decides what a person is *entitled* to. This decides whether they
//! are that person at all. Until now selecting an account was an assertion; with
//! this it is a claim that has to be backed.
//!
//! ## What is stored
//!
//! Only an Argon2id hash, never the password. Argon2id is memory-hard, so a
//! stolen database cannot be attacked at the rate a fast hash like SHA-256 would
//! allow — which matters here because the database sits on a machine inside a
//! plant, not behind a cloud provider's controls.
//!
//! Each password gets its own random salt, so two people choosing the same
//! password produce different hashes and neither fact is visible from the table.
//!
//! ## What is deliberately not done
//!
//! No password recovery. There is no email to send a reset to on an air-gapped
//! machine, and a recovery question is just a weaker second password. An
//! administrator resets another account; a lost administrator password is
//! recovered through the local procedure PS step 7 asks the site to define,
//! which is an operational answer rather than a software one.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use rand_core::OsRng;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

/// Minimum password length.
///
/// Length is required; composition is not. Rules demanding a symbol and a digit
/// reliably produce `Password1!` and a sticky note, which is worse than a long
/// passphrase. This follows the modern guidance of asking for length instead.
const MIN_PASSWORD_LENGTH: usize = 12;

/// Why a password was rejected, phrased for the person choosing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PasswordRejection {
    TooShort { minimum: usize, actual: usize },
    /// Entirely whitespace, or empty once trimmed.
    Blank,
}

impl PasswordRejection {
    pub fn reason(&self) -> String {
        match self {
            PasswordRejection::TooShort { minimum, actual } => format!(
                "That password is {actual} characters. It needs at least {minimum}. \
                 A memorable phrase of several words is easier to type and harder to guess \
                 than a short password with symbols in it."
            ),
            PasswordRejection::Blank => "A password cannot be blank.".to_string(),
        }
    }
}

/// Checks a candidate password against the policy.
pub fn check_password_policy(password: &str) -> Result<(), PasswordRejection> {
    if password.trim().is_empty() {
        return Err(PasswordRejection::Blank);
    }
    // Counted in characters rather than bytes, so a passphrase in an Indic
    // script is not rejected for being "short" when it is nothing of the kind.
    let length = password.chars().count();
    if length < MIN_PASSWORD_LENGTH {
        return Err(PasswordRejection::TooShort {
            minimum: MIN_PASSWORD_LENGTH,
            actual: length,
        });
    }
    Ok(())
}

/// Where sign-in stands for this deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AuthenticationStatus {
    /// No account has a password yet. The first administrator must set one
    /// before anything else can happen.
    AwaitingFirstAdministrator,
    /// At least one account is set up; sign-in is required.
    Configured,
}

pub struct CredentialStore {
    conn: Arc<Mutex<Connection>>,
}

impl CredentialStore {
    pub fn open(app_data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(app_data_dir)?;
        let conn = Connection::open(app_data_dir.join("sarathi.db"))
            .context("could not open the credential store")?;
        Self::from_connection(conn)
    }

    /// Builds a credential store against an arbitrary open connection.
    /// `pub(crate)` because only the store itself and its tests should
    /// initialise the schema; production code goes through [`Self::open`]
    /// which picks a path under the app data directory.
    pub(crate) fn from_connection(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS account_credentials (
                user_id    TEXT PRIMARY KEY,
                phc        TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );",
        )
        .context("could not prepare the credential schema")?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Whether anybody has a password yet.
    pub fn status(&self) -> Result<AuthenticationStatus> {
        let conn = self.conn.lock().expect("credential lock poisoned");
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM account_credentials", [], |r| r.get(0))?;
        Ok(if count == 0 {
            AuthenticationStatus::AwaitingFirstAdministrator
        } else {
            AuthenticationStatus::Configured
        })
    }

    pub fn has_password(&self, user_id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("credential lock poisoned");
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM account_credentials WHERE user_id = ?1",
            [user_id],
            |r| r.get(0),
        )?;
        Ok(count > 0)
    }

    /// Sets or replaces one account's password.
    ///
    /// The policy is checked here rather than only at the UI, so a caller that
    /// skips the form cannot store a weak password.
    pub fn set_password(&self, user_id: &str, password: &str) -> Result<()> {
        check_password_policy(password).map_err(|r| anyhow::anyhow!(r.reason()))?;

        let salt = SaltString::generate(&mut OsRng);
        let phc = Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("could not hash the password: {e}"))?
            .to_string();

        let conn = self.conn.lock().expect("credential lock poisoned");
        conn.execute(
            "INSERT INTO account_credentials (user_id, phc, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(user_id) DO UPDATE SET phc = excluded.phc, updated_at = excluded.updated_at",
            params![user_id, phc, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Checks a password against the stored hash.
    ///
    /// An unknown account still pays the cost of a verification against a dummy
    /// hash. Returning early would make "no such user" measurably faster than
    /// "wrong password", which is enough to enumerate who has an account.
    pub fn verify(&self, user_id: &str, password: &str) -> Result<bool> {
        let stored: Option<String> = {
            let conn = self.conn.lock().expect("credential lock poisoned");
            conn.query_row(
                "SELECT phc FROM account_credentials WHERE user_id = ?1",
                [user_id],
                |r| r.get(0),
            )
            .ok()
        };

        let phc = match &stored {
            Some(phc) => phc.as_str(),
            None => {
                // A real Argon2id hash of a value nobody holds, so the work done
                // on this path matches the work done on the success path.
                const DUMMY: &str = "$argon2id$v=19$m=19456,t=2,p=1$\
                                     c29tZXNhbHRzb21lc2FsdA$\
                                     8Yv1a1RmVYpqIYh1ZQjV0m0R1Q0m0R1Q0m0R1Q0m0R0";
                if let Ok(parsed) = PasswordHash::new(DUMMY) {
                    let _ = Argon2::default().verify_password(password.as_bytes(), &parsed);
                }
                return Ok(false);
            }
        };

        let parsed = PasswordHash::new(phc)
            .map_err(|e| anyhow::anyhow!("stored credential for {user_id} is unreadable: {e}"))?;

        Ok(Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> CredentialStore {
        CredentialStore::from_connection(Connection::open_in_memory().unwrap()).unwrap()
    }

    #[test]
    fn a_fresh_deployment_is_awaiting_its_first_administrator() {
        let store = store();
        assert_eq!(
            store.status().unwrap(),
            AuthenticationStatus::AwaitingFirstAdministrator
        );
    }

    #[test]
    fn setting_a_password_configures_the_deployment() {
        let store = store();
        store.set_password("admin", "correct horse battery").unwrap();
        assert_eq!(store.status().unwrap(), AuthenticationStatus::Configured);
        assert!(store.has_password("admin").unwrap());
    }

    #[test]
    fn the_right_password_verifies_and_a_wrong_one_does_not() {
        let store = store();
        store.set_password("admin", "correct horse battery").unwrap();
        assert!(store.verify("admin", "correct horse battery").unwrap());
        assert!(!store.verify("admin", "Correct horse battery").unwrap());
        assert!(!store.verify("admin", "something else entirely").unwrap());
    }

    #[test]
    fn an_unknown_account_never_verifies() {
        let store = store();
        store.set_password("admin", "correct horse battery").unwrap();
        assert!(!store.verify("ghost", "correct horse battery").unwrap());
    }

    /// The password is never recoverable from what is stored.
    #[test]
    fn the_password_itself_is_not_in_the_stored_record() {
        let store = store();
        let password = "correct horse battery staple";
        store.set_password("admin", password).unwrap();

        let conn = store.conn.lock().unwrap();
        let phc: String = conn
            .query_row("SELECT phc FROM account_credentials WHERE user_id = 'admin'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(!phc.contains(password));
        assert!(phc.starts_with("$argon2id$"), "{phc}");
    }

    /// Same password, two accounts, two different hashes — the salt is per-password.
    #[test]
    fn identical_passwords_produce_different_hashes() {
        let store = store();
        store.set_password("one", "correct horse battery").unwrap();
        store.set_password("two", "correct horse battery").unwrap();

        let conn = store.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT phc FROM account_credentials ORDER BY user_id").unwrap();
        let hashes: Vec<String> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(hashes.len(), 2);
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn changing_a_password_invalidates_the_old_one() {
        let store = store();
        store.set_password("admin", "the first passphrase").unwrap();
        store.set_password("admin", "the second passphrase").unwrap();
        assert!(!store.verify("admin", "the first passphrase").unwrap());
        assert!(store.verify("admin", "the second passphrase").unwrap());
    }

    #[test]
    fn short_and_blank_passwords_are_refused() {
        assert!(matches!(
            check_password_policy("short"),
            Err(PasswordRejection::TooShort { .. })
        ));
        assert!(matches!(check_password_policy("   "), Err(PasswordRejection::Blank)));
        assert!(check_password_policy("a long enough passphrase").is_ok());
    }

    /// The policy is enforced in the store, not just in the form.
    #[test]
    fn the_store_refuses_a_weak_password_even_if_the_ui_did_not() {
        let store = store();
        assert!(store.set_password("admin", "abc").is_err());
        assert_eq!(
            store.status().unwrap(),
            AuthenticationStatus::AwaitingFirstAdministrator
        );
    }

    /// Length is counted in characters, so a short-looking multi-byte passphrase
    /// is judged by what a person typed rather than by its encoded size.
    #[test]
    fn password_length_is_measured_in_characters() {
        // 12 Devanagari characters: long enough, though far more than 12 bytes.
        assert!(check_password_policy("आजकलमेरेपास").is_err(), "11 characters is short");
        assert!(check_password_policy("आजकलमेरेपासक").is_ok(), "12 characters is enough");
    }
}
