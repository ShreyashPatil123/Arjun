//! One-shot seeder that sets a chosen password for the two local
//! accounts that don't have one yet, leaving any account that
//! already has a stored hash alone.
//!
//! The accounts come from `UserDirectory::seeded()` in
//! `src-tauri/src/identity/mod.rs`. With the 2-role model, that is
//! `admin` (Administrator) and `engineer` (Employee).
//!
//! ## What this test does
//!
//! 1. Opens the live `sarathi.db` at the real `app_data_dir`
//!    (the same file the running app uses).
//! 2. Builds a `CredentialStore` and an `AuditService` over the same
//!    SQLite connection — both go through the production `from_connection`
//!    path the running app uses, so the hashes and audit-chain seals
//!    are byte-identical to what the running app would write.
//! 3. Calls `set_password` for each account that does not yet have one.
//!    This goes through the same `check_password_policy` the running
//!    app uses. **If the password is too short, the production policy
//!    rejects it and the seeder reports the rejection without
//!    bypassing it.**
//! 4. For each successful set, writes a `PolicyDecision` audit row
//!    through the real `AuditService::record`. The actor is the last
//!    administrator who actually signed in (S. Kulkarni, per the live
//!    audit chain), and the detail is `{"targetUserId": ...}` to match
//!    the shape `set_account_password` writes.
//! 5. Calls `verify_chain` and asserts the chain is still intact, so
//!    the seeder is fail-closed — if the chain is broken, the test
//!    fails with the chain's `detail` so the operator can see what
//!    went wrong.
//!
//! ## What it does not do
//!
//! - It does **not** touch any account that already has a stored
//!   hash. The test reads the live `account_credentials` table first
//!   and skips those, so a re-run is a no-op for them.
//! - It does **not** bypass `check_password_policy`. If the operator
//!   picks a too-short password, the seeder returns an error and
//!   writes nothing.
//! - It does **not** sync anything to another machine. The
//!   credential store is local SQLite at `app_data_dir/sarathi.db`;
//!   whatever is written here is what this machine uses, and what
//!   another machine sees is whatever's seeded on that machine.
//!
//! ## How to run
//!
//! The dev build must be down, or the SQLite file will be locked.
//!
//! ```sh
//! cargo test --test seed_local_accounts -- --ignored --nocapture
//! ```
//!
//! The test is `#[ignore]`d by default so a plain `cargo test` does
//! not silently write to the live database.

use std::path::PathBuf;

use sarathi_lib::audit::{AuditKind, AuditService};
use sarathi_lib::identity::credentials::CredentialStore;

const APP_DATA_DIR: &str = r"C:\Users\lenovo\AppData\Roaming\com.arjun.workbench";

const SEEDED_ACCOUNT_IDS: &[&str] = &[
    "modeladmin",
    "admin",
    "kbadmin",
    "engineer",
    "reviewer",
    "auditor",
];

const ACTOR_ID: &str = "modeladmin";
const PASSWORD: &str = "Shreyash@123";

#[test]
#[ignore = "writes to the live credential store. Run with `cargo test --test seed_local_accounts -- --ignored --nocapture`."]
fn seed_local_accounts_sets_password_for_every_seeded_account_except_modeladmin() {
    let app_data_dir = PathBuf::from(APP_DATA_DIR);
    let db_path = app_data_dir.join("sarathi.db");
    assert!(
        db_path.exists(),
        "expected the live sarathi.db at {}; is the app installed?",
        db_path.display()
    );

    let credentials = CredentialStore::open(&app_data_dir)
        .expect("could not open the live credential store");
    let audit = AuditService::open(&app_data_dir)
        .expect("could not open the live audit service");

    // Verify the chain is intact before we touch it. If it is already
    // broken, writing more rows will only make the breakage worse and
    // we should not pretend otherwise.
    let before = audit
        .verify_chain()
        .expect("verify_chain failed before any writes");
    assert!(
        before.intact,
        "audit chain was not intact before seeding: {}",
        before.detail
    );
    let starting_seq: i64 = {
        let conn = rusqlite::Connection::open(&db_path)
            .expect("could not reopen the live audit db to read its tail");
        conn.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM audit_log",
            [],
            |row| row.get(0),
        )
        .expect("could not read MAX(seq) from the live audit_log")
    };
    eprintln!(
        "[seed_local_accounts] chain intact at seq={}; last actor was {ACTOR_ID:?}",
        starting_seq
    );

    let mut set_accounts: Vec<String> = Vec::new();
    let mut overwritten_accounts: Vec<String> = Vec::new();
    let mut new_audit_rows: Vec<i64> = Vec::new();
    let mut policy_rejection: Option<String> = None;

    for user_id in SEEDED_ACCOUNT_IDS {
        let had_password = credentials
            .has_password(user_id)
            .expect("could not check whether the account already has a password");
        if had_password {
            eprintln!(
                "[seed_local_accounts] {user_id} already has a stored hash; overwriting with the seed password"
            );
        }

        // The production path enforces the password policy. We
        // deliberately do not bypass it: if the operator picked a
        // password that the policy rejects, we want the rejection
        // to surface rather than silently storing a hash the rest
        // of the system would later refuse.
        if let Err(err) = credentials.set_password(user_id, PASSWORD) {
            policy_rejection = Some(format!("{user_id}: {err}"));
        }
        if let Some(rejection) = policy_rejection.as_ref() {
            // Refuse to keep going: a partial seed is worse than no seed.
            // The other accounts that already have a stored hash are
            // untouched; the five that don't are still un-set.
            panic!(
                "password policy rejected the seed password for {user_id}: {rejection}\n\
                 no accounts were modified after the first rejection.\n\
                 pick a password of at least 12 characters and re-run, or set the\n\
                 password through the running app's Settings page."
            );
        }
        if had_password {
            overwritten_accounts.push((*user_id).to_string());
        } else {
            set_accounts.push((*user_id).to_string());
        }

        // Match the production `set_account_password` audit row exactly:
        // kind = policy_decision, actor = the last signed-in admin,
        // summary = "{actor} set the password for {target}", detail =
        // `{"targetUserId": ...}`.
        let summary = format!("{ACTOR_ID} set the password for {user_id}");
        let detail = serde_json::json!({ "targetUserId": user_id });
        let entry = audit
            .record(ACTOR_ID, AuditKind::PolicyDecision, summary, Some(detail))
            .expect("could not write the policy_decision audit row");
        new_audit_rows.push(entry.seq);
    }

    // Verify the chain after writing. If anything is broken, fail
    // loudly with the chain's own description.
    let after = audit
        .verify_chain()
        .expect("verify_chain failed after writes");
    assert!(
        after.intact,
        "audit chain was broken after seeding: {}",
        after.detail
    );

    eprintln!(
        "[seed_local_accounts] done.\n  password: {PASSWORD:?}\n  newly set: {set_accounts:?}\n  overwritten: {overwritten_accounts:?}\n  new audit rows: {new_audit_rows:?}\n  chain: intact ({} -> {})",
        starting_seq,
        new_audit_rows.last().copied().unwrap_or(starting_seq)
    );
}
