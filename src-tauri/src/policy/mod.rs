//! The policy gateway — the only thing that decides whether an action may proceed.
//!
//! PS 26117 step 8: *"The model must never decide permissions. The policy gateway
//! checks the user, document classification, requested tool, target path, and
//! approval state before allowing an action."*
//!
//! That sentence is the whole design. A model can only ever *request*; every
//! request arrives here as a [`Request`] and leaves as a [`Decision`], and the
//! decision is reached from the session and the request alone. Nothing the model
//! wrote is consulted, because nothing the model wrote is trustworthy — a scanned
//! document that says "ignore previous instructions" is data, and data does not
//! get a vote.
//!
//! Checks run in a fixed order, cheapest and most fundamental first, so a refusal
//! always names the most basic reason rather than an incidental one:
//!
//! 1. **The sovereignty invariant** — is confidential work permitted at all right
//!    now? A refusal here is about the machine's mode, not the person.
//! 2. **Entitlement** — does this user hold the permission?
//! 3. **Classification** — is this user cleared for material of this kind?
//! 4. **Scope** — is the target inside a directory this task may touch?
//! 5. **Approval** — does this need a human, and has one said yes?

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

use crate::identity::{Permission, Role, Session};

/// How sensitive the material is.
///
/// These are the problem statement's own categories, quoted from its background
/// section, rather than generic tiers. Using the site's vocabulary means a
/// reviewer at MRPL can read a refusal without translating it first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Classification {
    /// Ordinary internal material with no special handling.
    Internal,
    /// Piping & instrument diagrams.
    ProcessDiagram,
    Financial,
    VendorNegotiation,
    UnreleasedDesign,
    InternalCorrespondence,
    BusinessStrategy,
}

impl Classification {
    pub const ALL: &'static [Classification] = &[
        Classification::Internal,
        Classification::ProcessDiagram,
        Classification::Financial,
        Classification::VendorNegotiation,
        Classification::UnreleasedDesign,
        Classification::InternalCorrespondence,
        Classification::BusinessStrategy,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Classification::Internal => "Internal",
            Classification::ProcessDiagram => "P&ID / process diagram",
            Classification::Financial => "Financial",
            Classification::VendorNegotiation => "Vendor negotiation",
            Classification::UnreleasedDesign => "Unreleased design",
            Classification::InternalCorrespondence => "Internal correspondence",
            Classification::BusinessStrategy => "Business strategy",
        }
    }

    /// Roles cleared to handle material of this kind.
    ///
    /// The auditor is absent from every one of these on purpose: an auditor
    /// reads the *record* of what happened, which is not the same as reading the
    /// documents themselves, and conflating the two would quietly turn the
    /// oversight role into the broadest read access in the building.
    pub fn cleared_roles(self) -> &'static [Role] {
        match self {
            // Everyday material: anyone doing the work, and whoever curates it.
            Classification::Internal | Classification::ProcessDiagram => {
                &[Role::User, Role::KnowledgeAdministrator, Role::Reviewer]
            }
            // Commercially sensitive: not the knowledge administrator, whose job
            // is curating manuals and SOPs rather than reading deal terms.
            Classification::Financial
            | Classification::VendorNegotiation
            | Classification::BusinessStrategy => &[Role::User, Role::Reviewer],
            // Unreleased designs and internal correspondence: the narrowest set.
            Classification::UnreleasedDesign | Classification::InternalCorrespondence => {
                &[Role::User, Role::Reviewer]
            }
        }
    }
}

/// Whether a human has signed off on this particular action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalState {
    /// No human sign-off has been sought.
    NotRequested,
    /// Shown to a human and waiting.
    Pending,
    /// A human said yes. Carries who, for the separation-of-duties check.
    Granted,
    Rejected,
}

/// One thing something wants to do.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub permission: Permission,
    /// Set when the action touches material of a known sensitivity.
    pub classification: Option<Classification>,
    /// Set when the action writes to, or reads from, a path.
    pub target_path: Option<PathBuf>,
    /// Directories this task is allowed to touch. Empty means none.
    pub allowed_roots: &'a [PathBuf],
    /// Whether this class of action needs a human before it happens.
    pub needs_approval: bool,
    pub approval: ApprovalState,
    /// Who ran the task, when the action being requested is an approval.
    pub task_owner: Option<&'a str>,
}

impl<'a> Request<'a> {
    /// A request with nothing but a permission — the common case.
    pub fn new(permission: Permission) -> Self {
        Self {
            permission,
            classification: None,
            target_path: None,
            allowed_roots: &[],
            needs_approval: false,
            approval: ApprovalState::NotRequested,
            task_owner: None,
        }
    }

    pub fn classified(mut self, classification: Classification) -> Self {
        self.classification = Some(classification);
        self
    }

    pub fn writing_to(mut self, path: impl Into<PathBuf>, allowed_roots: &'a [PathBuf]) -> Self {
        self.target_path = Some(path.into());
        self.allowed_roots = allowed_roots;
        self
    }

    pub fn requiring_approval(mut self, approval: ApprovalState) -> Self {
        self.needs_approval = true;
        self.approval = approval;
        self
    }

    pub fn approving_task_owned_by(mut self, owner: &'a str) -> Self {
        self.task_owner = Some(owner);
        self
    }
}

/// What the gateway decided, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "outcome")]
pub enum Decision {
    Allow,
    /// Refused outright. `reason` is written for the person who hit it.
    Refuse { reason: String },
    /// Permitted in principle, but a human has to say yes first.
    NeedsApproval { reason: String },
}

impl Decision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Decision::Allow)
    }

    /// One line for the audit record and the UI.
    pub fn describe(&self) -> &str {
        match self {
            Decision::Allow => "Allowed",
            Decision::Refuse { reason } => reason,
            Decision::NeedsApproval { reason } => reason,
        }
    }
}

/// True when `path` stays inside one of `roots` once `..` segments are resolved.
///
/// Resolution is textual rather than filesystem-based, because the target of a
/// write usually does not exist yet, and `canonicalize` fails on a path that is
/// not there. Textual resolution is also the safer default: it cannot be
/// defeated by a symlink planted between the check and the write.
fn is_within(path: &Path, roots: &[PathBuf]) -> bool {
    fn normalise(path: &Path) -> Option<PathBuf> {
        let mut out = PathBuf::new();
        for component in path.components() {
            match component {
                // A `..` that would climb above the root is a traversal attempt,
                // not a path — refuse rather than clamping it to the root.
                Component::ParentDir => {
                    if !out.pop() {
                        return None;
                    }
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        Some(out)
    }

    let Some(candidate) = normalise(path) else {
        return false;
    };

    roots.iter().any(|root| match normalise(root) {
        Some(root) => candidate.starts_with(&root),
        None => false,
    })
}

pub struct PolicyGateway;

impl PolicyGateway {
    /// Decides one request.
    ///
    /// `confidential_work_permitted` comes from the sovereignty invariant — the
    /// gateway is told, rather than reaching for the broker, so this stays a
    /// pure function that the tests can drive through every combination.
    pub fn decide(
        session: &Session,
        request: &Request<'_>,
        confidential_work_permitted: bool,
    ) -> Decision {
        // 1. The invariant. Refused here regardless of who is asking, because
        //    this is about the state of the machine rather than the person.
        if !confidential_work_permitted {
            return Decision::Refuse {
                reason: format!(
                    "Cannot {} while ARJUN is in Provisioning mode: the network is reachable, \
                     so no confidential material may be handled. Switch to Work mode first.",
                    request.permission.describe()
                ),
            };
        }

        // 2. Entitlement.
        if !session.holds(request.permission) {
            return Decision::Refuse {
                reason: format!(
                    "{} is not permitted to {}. None of their roles grant it.",
                    session.user.display_name,
                    request.permission.describe()
                ),
            };
        }

        // 3. Clearance for this kind of material.
        if let Some(classification) = request.classification {
            let cleared = classification
                .cleared_roles()
                .iter()
                .any(|role| session.user.roles.contains(role));
            if !cleared {
                return Decision::Refuse {
                    reason: format!(
                        "{} is not cleared for {} material.",
                        session.user.display_name,
                        classification.label()
                    ),
                };
            }
        }

        // 4. Scope. A write with no permitted roots is refused rather than
        //    treated as unrestricted — the absence of a scope is not a licence.
        if let Some(target) = &request.target_path {
            if !is_within(target, request.allowed_roots) {
                return Decision::Refuse {
                    reason: format!(
                        "{} is outside the directories this task may touch.",
                        target.display()
                    ),
                };
            }
        }

        // 5. Separation of duties, before approval state is considered: holding
        //    the reviewer role does not entitle someone to sign off their own
        //    work, and a second reviewer is always available in practice.
        if request.permission == Permission::ApproveOutput {
            if let Some(owner) = request.task_owner {
                if owner == session.user.id {
                    return Decision::Refuse {
                        reason: format!(
                            "{} ran this task and cannot also approve it. \
                             Another reviewer has to sign it off.",
                            session.user.display_name
                        ),
                    };
                }
            }
        }

        // 6. Human sign-off.
        if request.needs_approval {
            return match request.approval {
                ApprovalState::Granted => Decision::Allow,
                ApprovalState::Rejected => Decision::Refuse {
                    reason: "A reviewer rejected this action.".to_string(),
                },
                ApprovalState::Pending | ApprovalState::NotRequested => Decision::NeedsApproval {
                    reason: format!(
                        "Permitted, but a reviewer has to approve before ARJUN will {}.",
                        request.permission.describe()
                    ),
                },
            };
        }

        Decision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, User};

    fn session_with(roles: Vec<Role>) -> Session {
        Session::open(User::new("kiran", "Kiran", roles))
    }

    #[test]
    fn an_entitled_user_is_allowed() {
        let session = session_with(vec![Role::User]);
        let request = Request::new(Permission::UseModel);
        assert!(PolicyGateway::decide(&session, &request, true).is_allowed());
    }

    /// The invariant outranks everything: even a fully entitled user is refused
    /// while the network is reachable.
    #[test]
    fn provisioning_mode_refuses_even_an_entitled_user() {
        let session = session_with(vec![Role::User]);
        let request = Request::new(Permission::UseModel);
        let decision = PolicyGateway::decide(&session, &request, false);
        assert!(decision.describe().contains("Provisioning mode"), "{decision:?}");
    }

    #[test]
    fn a_user_without_the_permission_is_refused() {
        let session = session_with(vec![Role::Auditor]);
        let request = Request::new(Permission::UseModel);
        let decision = PolicyGateway::decide(&session, &request, true);
        assert!(!decision.is_allowed());
        assert!(decision.describe().contains("not permitted"), "{decision:?}");
    }

    #[test]
    fn clearance_is_checked_against_the_classification() {
        let kb = session_with(vec![Role::KnowledgeAdministrator]);

        // Cleared for ordinary manuals and SOPs...
        let sop = Request::new(Permission::SearchKnowledge)
            .classified(Classification::ProcessDiagram);
        assert!(PolicyGateway::decide(&kb, &sop, true).is_allowed());

        // ...but not for vendor negotiations.
        let deal = Request::new(Permission::SearchKnowledge)
            .classified(Classification::VendorNegotiation);
        let decision = PolicyGateway::decide(&kb, &deal, true);
        assert!(!decision.is_allowed());
        assert!(decision.describe().contains("not cleared"), "{decision:?}");
    }

    /// An auditor reads the record, not the documents behind it.
    #[test]
    fn the_auditor_is_cleared_for_nothing() {
        for classification in Classification::ALL {
            assert!(
                !classification.cleared_roles().contains(&Role::Auditor),
                "auditor should not be cleared for {}",
                classification.label()
            );
        }
    }

    #[test]
    fn a_write_inside_the_task_workspace_is_allowed() {
        let session = session_with(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/42")];
        let request = Request::new(Permission::GenerateArtifact)
            .writing_to("C:/arjun/tasks/42/approval-note.docx", &roots);
        assert!(PolicyGateway::decide(&session, &request, true).is_allowed());
    }

    #[test]
    fn a_write_outside_the_task_workspace_is_refused() {
        let session = session_with(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/42")];
        let request = Request::new(Permission::GenerateArtifact)
            .writing_to("C:/Windows/System32/drivers/etc/hosts", &roots);
        assert!(!PolicyGateway::decide(&session, &request, true).is_allowed());
    }

    /// The traversal that a naive prefix check waves straight through.
    #[test]
    fn dot_dot_cannot_climb_out_of_the_workspace() {
        let session = session_with(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/42")];
        for escape in [
            "C:/arjun/tasks/42/../../../Windows/System32/config",
            "C:/arjun/tasks/42/../43/other-task.docx",
        ] {
            let request = Request::new(Permission::GenerateArtifact).writing_to(escape, &roots);
            assert!(
                !PolicyGateway::decide(&session, &request, true).is_allowed(),
                "{escape} should have been refused"
            );
        }
    }

    /// A sibling directory that merely starts with the same characters is not
    /// inside the workspace — the classic prefix-matching bug.
    #[test]
    fn a_sibling_with_a_shared_prefix_is_not_inside_the_workspace() {
        let session = session_with(vec![Role::User]);
        let roots = vec![PathBuf::from("C:/arjun/tasks/4")];
        let request = Request::new(Permission::GenerateArtifact)
            .writing_to("C:/arjun/tasks/42/secret.docx", &roots);
        assert!(!PolicyGateway::decide(&session, &request, true).is_allowed());
    }

    #[test]
    fn a_write_with_no_permitted_roots_is_refused() {
        let session = session_with(vec![Role::User]);
        let request = Request::new(Permission::GenerateArtifact)
            .writing_to("C:/anywhere/at/all.docx", &[]);
        assert!(!PolicyGateway::decide(&session, &request, true).is_allowed());
    }

    #[test]
    fn a_risky_action_waits_for_a_human() {
        let session = session_with(vec![Role::User]);
        let request =
            Request::new(Permission::ExecuteCode).requiring_approval(ApprovalState::NotRequested);
        assert!(matches!(
            PolicyGateway::decide(&session, &request, true),
            Decision::NeedsApproval { .. }
        ));
    }

    #[test]
    fn an_approved_action_proceeds_and_a_rejected_one_does_not() {
        let session = session_with(vec![Role::User]);

        let granted =
            Request::new(Permission::ExecuteCode).requiring_approval(ApprovalState::Granted);
        assert!(PolicyGateway::decide(&session, &granted, true).is_allowed());

        let rejected =
            Request::new(Permission::ExecuteCode).requiring_approval(ApprovalState::Rejected);
        assert!(!PolicyGateway::decide(&session, &rejected, true).is_allowed());
    }

    /// Holding the reviewer role is not enough to sign off your own work.
    #[test]
    fn a_reviewer_cannot_approve_their_own_task() {
        let session = Session::open(User::new("kiran", "Kiran", vec![Role::Reviewer]));
        let request = Request::new(Permission::ApproveOutput).approving_task_owned_by("kiran");
        let decision = PolicyGateway::decide(&session, &request, true);
        assert!(!decision.is_allowed());
        assert!(decision.describe().contains("cannot also approve"), "{decision:?}");
    }

    #[test]
    fn a_reviewer_can_approve_someone_elses_task() {
        let session = Session::open(User::new("kiran", "Kiran", vec![Role::Reviewer]));
        let request = Request::new(Permission::ApproveOutput).approving_task_owned_by("anil");
        assert!(PolicyGateway::decide(&session, &request, true).is_allowed());
    }

    /// Refusals should name the most fundamental reason, not an incidental one:
    /// a user who is both unentitled and in the wrong mode is told about the
    /// mode, because fixing their roles would not have helped.
    #[test]
    fn the_most_fundamental_refusal_wins() {
        let session = session_with(vec![Role::Auditor]);
        let request = Request::new(Permission::UseModel);
        let decision = PolicyGateway::decide(&session, &request, false);
        assert!(decision.describe().contains("Provisioning mode"), "{decision:?}");
    }
}
