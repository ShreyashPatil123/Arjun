//! Whether this installation can still record what it does.
//!
//! ## Why this exists
//!
//! ARJUN's whole claim is that every run leaves a checkable record. Two stores
//! carry it: the append-only task event log (`events::TaskEventLog`) and the
//! per-run task record on disk (`tasks::save`).
//!
//! Both used to fail open. If the database could not be opened, start-up caught
//! the error, logged it, and substituted an **in-memory** log — so the
//! application came up looking normal, ran tasks, wrote files, and kept a
//! history that evaporated when the process exited. If a write failed
//! mid-run, four of the five call sites logged a warning and carried on. In
//! both cases the product went on making the claim it could no longer support,
//! and nobody using it was told.
//!
//! The comment justifying the in-memory fallback said "an unrecorded run is
//! worse than a recorded one and better than an application that will not
//! start". That is the right trade for *opening the window* and the wrong one
//! for *doing work*. So the two are now separated:
//!
//! - **The desktop opens.** A person can read what is already there, look at
//!   past runs, change settings, and find out what is wrong.
//! - **Runs do not start, and side-effecting tools do not run.** Anything that
//!   would produce a record there is no longer anywhere to put is refused,
//!   with the reason.
//!
//! ## Why writes flip it too
//!
//! A log that opened successfully can stop being writable — a full disk, a
//! revoked permission, a file locked by a backup agent. A health state fixed at
//! start-up would say "durable" for the rest of a session in which nothing was
//! being written. So a storage failure reported by either store degrades this,
//! and the next run is refused rather than the next thousand being silently
//! unrecorded.
//!
//! ## What is *not* here
//!
//! Idempotent outcomes are not failures. An event refused because it is already
//! in the log, or because the run already has an ending, is the append doing
//! its job — see [`crate::agent_runtime::events::AppendError`]. Only
//! `Storage` degrades this.

use std::sync::RwLock;

use serde::{Deserialize, Serialize};

/// How the audit stores are doing, as the UI is told it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// `rename_all` renames the variants; `rename_all_fields` renames what is
// inside them. Both are needed for the UI to see `atStartup`.
#[serde(tag = "state", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum AuditState {
    /// Both stores are on disk and writing. Runs may start.
    Durable,
    /// Something a run's record depends on is not working.
    ///
    /// The application is usable read-only: history opens, settings open,
    /// nothing that would need recording is allowed to start.
    Degraded {
        /// What went wrong, in the sentence shown to a person.
        because: String,
        /// Whether the failure was at start-up (the store never opened) rather
        /// than during a run (a write failed). The remedies differ and the
        /// message says which.
        at_startup: bool,
    },
}

impl AuditState {
    pub const fn is_durable(&self) -> bool {
        matches!(self, AuditState::Durable)
    }
}

/// The live health of this installation's audit storage.
///
/// One instance, managed by the application and shared with everything that
/// writes a record. Interior mutability rather than a fresh value per read:
/// a write failing in one run has to be visible to the next one.
#[derive(Debug)]
pub struct AuditHealth {
    state: RwLock<AuditState>,
}

impl Default for AuditHealth {
    fn default() -> Self {
        Self::durable()
    }
}

impl AuditHealth {
    /// The ordinary case: both stores opened and are writable.
    pub fn durable() -> Self {
        Self {
            state: RwLock::new(AuditState::Durable),
        }
    }

    /// Degraded from the moment the application started.
    pub fn degraded_at_startup(because: impl Into<String>) -> Self {
        Self {
            state: RwLock::new(AuditState::Degraded {
                because: because.into(),
                at_startup: true,
            }),
        }
    }

    /// What to show, and what to decide on.
    ///
    /// A poisoned lock reads as degraded. The alternative is to report health
    /// this cannot actually observe, which is the failure mode the whole module
    /// exists to remove.
    pub fn state(&self) -> AuditState {
        match self.state.read() {
            Ok(state) => state.clone(),
            Err(_) => AuditState::Degraded {
                because: "The audit health record itself could not be read.".to_string(),
                at_startup: false,
            },
        }
    }

    pub fn is_durable(&self) -> bool {
        self.state().is_durable()
    }

    /// Records that a durable write failed, degrading this installation.
    ///
    /// First failure wins. The first thing that broke is the useful one to
    /// show; the twenty consequences of it are not, and a message that keeps
    /// being overwritten reads as though the fault is moving around.
    pub fn writes_failed(&self, because: impl Into<String>) {
        let Ok(mut state) = self.state.write() else {
            return;
        };
        if matches!(*state, AuditState::Degraded { .. }) {
            return;
        }
        *state = AuditState::Degraded {
            because: because.into(),
            at_startup: false,
        };
    }

    /// The sentence to refuse a run with, or `None` when there is no reason to.
    ///
    /// Phrased for the person who has to fix it: what is not working, and what
    /// the application is therefore not going to do.
    pub fn refusal(&self) -> Option<String> {
        match self.state() {
            AuditState::Durable => None,
            AuditState::Degraded {
                because,
                at_startup,
            } => Some(if at_startup {
                format!(
                    "This installation cannot record what it does, so it will not run tasks. \
                     {because} Past tasks can still be read. Nothing will run until the record \
                     can be written again."
                )
            } else {
                format!(
                    "Recording stopped working while this application was open, so no further \
                     tasks will be run. {because} Restart once the problem is fixed; what was \
                     recorded before it happened is still there."
                )
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_healthy_installation_refuses_nothing() {
        let health = AuditHealth::durable();
        assert!(health.is_durable());
        assert_eq!(health.refusal(), None);
    }

    #[test]
    fn a_log_that_never_opened_refuses_runs_and_says_why() {
        let health = AuditHealth::degraded_at_startup("The task event log is on a read-only disk.");
        assert!(!health.is_durable());
        let refusal = health.refusal().expect("a reason");
        assert!(refusal.contains("read-only disk"), "{refusal}");
        // The distinction a person acts on: the desktop is usable, the work is
        // not going to start.
        assert!(refusal.contains("can still be read"), "{refusal}");
    }

    #[test]
    fn a_write_that_fails_mid_session_degrades_the_installation() {
        let health = AuditHealth::durable();
        health.writes_failed("The disk is full.");
        assert!(!health.is_durable());
        let refusal = health.refusal().expect("a reason");
        assert!(refusal.contains("disk is full"), "{refusal}");
        assert!(refusal.contains("Restart"), "{refusal}");
    }

    #[test]
    fn the_first_failure_is_the_one_reported() {
        // The twenty consequences of a full disk are not twenty faults, and a
        // message that keeps being overwritten reads as though the problem is
        // moving around.
        let health = AuditHealth::durable();
        health.writes_failed("The disk is full.");
        health.writes_failed("The record could not be saved.");
        health.writes_failed("The ending could not be written.");
        let AuditState::Degraded { because, .. } = health.state() else {
            panic!("degraded");
        };
        assert_eq!(because, "The disk is full.");
    }

    #[test]
    fn a_startup_failure_is_not_overwritten_by_the_writes_it_causes() {
        let health = AuditHealth::degraded_at_startup("The database could not be opened.");
        health.writes_failed("The ending could not be written.");
        let AuditState::Degraded {
            because,
            at_startup,
        } = health.state()
        else {
            panic!("degraded");
        };
        assert_eq!(because, "The database could not be opened.");
        assert!(at_startup, "the original cause was at start-up");
    }

    #[test]
    fn the_state_serialises_for_the_ui() {
        let encoded = serde_json::to_value(AuditHealth::durable().state()).expect("serialises");
        assert_eq!(encoded["state"], "durable");

        let degraded = AuditHealth::degraded_at_startup("no disk").state();
        let encoded = serde_json::to_value(degraded).expect("serialises");
        assert_eq!(encoded["state"], "degraded");
        assert_eq!(encoded["because"], "no disk");
        assert_eq!(encoded["atStartup"], true);
    }
}
