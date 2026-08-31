//! Who is acting, and what that entitles them to.
//!
//! PS 26117 step 8 is explicit that the model must never decide permissions. So
//! entitlement is decided here, from the signed-in user's roles, and nothing in
//! this module can be reached by anything a model emits — it only ever sees a
//! [`Session`] the application already established.
//!
//! ## Two active roles
//!
//! Arjun recognises exactly two roles:
//!
//! - **`Administrator`** — full control. A person who runs the deployment,
//!   configures the system, manages models and accounts, and is the only one
//!   who can put ARJUN into Provisioning mode. A superset of every Employee
//!   permission, because there is no separation-of-duties argument for a
//!   two-role product: the Administrator is the operator of last resort.
//! - **`Employee`** — the normal end user. Chats with the local model, runs
//!   tasks through the orchestrator, uploads permitted documents, generates
//!   permitted artifacts. Cannot install, delete or load models, manage users,
//!   change policy, or enter Provisioning mode.
//!
//! ## Why the enum keeps more variants than the active role list
//!
//! The internal [`Role`] enum still carries the historical variants
//! (`ModelAdministrator`, `KnowledgeAdministrator`, `Reviewer`, `Auditor`)
//! because a large test surface in the rest of the crate refers to them by
//! name. They are not active roles: every grant table below returns `false`
//! for them, [`Role::ALL`] no longer lists them, and the seeded
//! [`UserDirectory`] does not offer them. New code should pick from
//! [`Role::Administrator`] or [`Role::Employee`].

pub mod credentials;

use serde::{Deserialize, Serialize};

pub use credentials::{
    check_password_policy, AuthenticationStatus, CredentialStore, PasswordRejection,
};

/// A distinct thing someone can be entitled to do.
///
/// Taken from PS step 8's own list, plus [`Permission::EnterProvisioning`] —
/// making the network reachable is the single most consequential action in the
/// product, so it is a permission in its own right rather than a side effect of
/// being an administrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Permission {
    /// Send a prompt to a local model.
    UseModel,
    /// Bring a document into the workbench.
    UploadDocument,
    /// Query the local knowledge base.
    SearchKnowledge,
    /// Run model-written code in the sandbox.
    ExecuteCode,
    /// Write a file outside the task workspace.
    WriteFiles,
    /// Produce a Word, Excel or PowerPoint deliverable.
    GenerateArtifact,
    /// Accept an output, or a proposed action awaiting a human.
    ApproveOutput,
    /// Register a new model in the local registry.
    ImportModel,
    /// Read the audit record.
    ViewAuditLog,
    /// Change roles, classifications or tool policy.
    ModifyPolicy,
    /// Switch ARJUN into Provisioning mode, making the network reachable.
    EnterProvisioning,
}

impl Permission {
    /// Wording used when the permission is refused, so the message names the
    /// action in the user's terms rather than echoing an enum variant.
    pub const fn describe(self) -> &'static str {
        match self {
            Permission::UseModel => "use a model",
            Permission::UploadDocument => "upload a document",
            Permission::SearchKnowledge => "search the knowledge base",
            Permission::ExecuteCode => "run code in the sandbox",
            Permission::WriteFiles => "write files outside the task workspace",
            Permission::GenerateArtifact => "generate a document",
            Permission::ApproveOutput => "approve an output",
            Permission::ImportModel => "import a model",
            Permission::ViewAuditLog => "view the audit log",
            Permission::ModifyPolicy => "change policy",
            Permission::EnterProvisioning => "put ARJUN into Provisioning mode",
        }
    }
}

/// A job someone does with the workbench.
///
/// The product supports exactly two active roles, [`Role::Administrator`] and
/// [`Role::Employee`]. The other variants exist for backwards compatibility
/// with a large body of test code and historical role names; they grant
/// nothing through [`Role::grants`], are not listed in [`Role::ACTIVE`], and
/// are not offered by the seeded [`UserDirectory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    /// Runs the deployment. Holds every permission — the superset.
    Administrator,
    /// The normal end user. Holds the everyday work permissions only.
    Employee,

    // ── Legacy variants kept for internal call sites and tests ──────────────
    // These do not grant any permission. They are not active roles. New code
    // must not produce them; new tests should use the two active roles above.
    #[doc(hidden)]
    ModelAdministrator,
    #[doc(hidden)]
    KnowledgeAdministrator,
    #[doc(hidden)]
    User,
    #[doc(hidden)]
    Reviewer,
    #[doc(hidden)]
    Auditor,
}

impl Role {
    /// The two active roles, in the order they are shown in the UI.
    pub const ACTIVE: &'static [Role] = &[Role::Administrator, Role::Employee];

    /// Every variant, including the legacy ones. Use [`Self::ACTIVE`] when the
    /// question is "what should the user be able to pick?"; this is the
    /// exhaustive list for permission-matrix tests.
    pub const ALL: &'static [Role] = &[
        Role::Administrator,
        Role::Employee,
        Role::ModelAdministrator,
        Role::KnowledgeAdministrator,
        Role::User,
        Role::Reviewer,
        Role::Auditor,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Role::Administrator => "Administrator",
            Role::Employee => "Employee",
            // Legacy labels are preserved so older test strings still parse.
            Role::ModelAdministrator => "Model administrator",
            Role::KnowledgeAdministrator => "Knowledge administrator",
            Role::User => "Employee",
            Role::Reviewer => "Reviewer",
            Role::Auditor => "Auditor",
        }
    }

    /// Whether this is one of the two active roles. Legacy variants return
    /// `false`; new code should use this when it needs to filter for "real"
    /// roles (e.g. the ARJUN dropdown).
    pub const fn is_active(self) -> bool {
        matches!(self, Role::Administrator | Role::Employee)
    }

    /// The two-role entitlement matrix, in one place.
    ///
    /// The matrix is the whole story: a permission is held if the role grants
    /// it. Administrator is the superset (every permission). Employee holds
    /// the everyday work permissions only — model interaction, document
    /// upload, knowledge search, sandboxed code execution, artifact
    /// generation, and approval. Legacy variants grant nothing.
    ///
    /// [`Permission::WriteFiles`] is intentionally not granted by either
    /// role: writes outside the task workspace are gated per-path by the
    /// policy gateway and always need approval, so a role that handed it
    /// out unconditionally would undercut that.
    pub fn grants(self, permission: Permission) -> bool {
        use Permission::*;
        match self {
            Role::Administrator => !matches!(permission, WriteFiles),
            Role::Employee => matches!(
                permission,
                UseModel
                    | UploadDocument
                    | SearchKnowledge
                    | ExecuteCode
                    | GenerateArtifact
                    | ApproveOutput
            ),
            // Legacy variants grant nothing in the active product.
            Role::ModelAdministrator
            | Role::KnowledgeAdministrator
            | Role::User
            | Role::Reviewer
            | Role::Auditor => false,
        }
    }
}

/// A local account.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct User {
    pub id: String,
    pub display_name: String,
    pub roles: Vec<Role>,
    /// Free text, shown in the audit record so a reviewer can tell who this was.
    pub department: Option<String>,
}

impl User {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>, roles: Vec<Role>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            roles,
            department: None,
        }
    }

    /// The single "headline" role for this account — Administrator or Employee
    /// — for display in the ARJUN account menu. Administrator wins; if no
    /// active role is held, Employee is the safe default.
    pub fn headline_role(&self) -> Role {
        if self.roles.iter().any(|r| matches!(r, Role::Administrator)) {
            Role::Administrator
        } else if self.roles.iter().any(|r| matches!(r, Role::Employee)) {
            Role::Employee
        } else {
            Role::Employee
        }
    }

    /// True when any held role grants the permission.
    ///
    /// A user with no roles holds nothing. That is the correct reading of an
    /// account that has been created but not yet given a job.
    pub fn holds(&self, permission: Permission) -> bool {
        self.roles.iter().any(|r| r.grants(permission))
    }

    /// Every permission this user holds, for display on an account screen.
    pub fn permissions(&self) -> Vec<Permission> {
        use Permission::*;
        [
            UseModel,
            UploadDocument,
            SearchKnowledge,
            ExecuteCode,
            WriteFiles,
            GenerateArtifact,
            ApproveOutput,
            ImportModel,
            ViewAuditLog,
            ModifyPolicy,
            EnterProvisioning,
        ]
        .into_iter()
        .filter(|p| self.holds(*p))
        .collect()
    }
}

/// The signed-in user for this run of the application.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub user: User,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl Session {
    pub fn open(user: User) -> Self {
        Self {
            user,
            started_at: chrono::Utc::now(),
        }
    }

    pub fn holds(&self, permission: Permission) -> bool {
        self.user.holds(permission)
    }
}


/// The local accounts this deployment knows about.
///
/// Seeded with one Administrator (S. Kulkarni) and five Employees so the
/// two-role model is demonstrable from first launch and the operator can
/// see every named person on the team under their correct role label. The
/// directory is the authoritative list of which display names map to
/// which role — the seeded table below is the only place the operator
/// assigns roles. An installation whose only account is an
/// all-powerful administrator teaches everyone to use that account, which
/// is the failure this design exists to avoid; an installation whose only
/// accounts are Employees teaches the same lesson in the other direction.
/// Exactly one of the six seeded accounts holds the Administrator role,
/// so the two halves of the matrix are both reachable from first launch.
pub struct UserDirectory {
    users: Vec<User>,
}

impl Default for UserDirectory {
    fn default() -> Self {
        Self::seeded()
    }
}

impl UserDirectory {
    /// The six demo accounts. Exactly one — S. Kulkarni — is the
    /// Administrator; the other five are Employees. Every account holds
    /// exactly one role, so the account selector shows a single label
    /// per person and the role list never leaks a legacy variant.
    pub fn seeded() -> Self {
        Self {
            users: vec![
                User {
                    id: "modeladmin".into(),
                    display_name: "S. Kulkarni".into(),
                    roles: vec![Role::Administrator],
                    department: Some("IT & Systems".into()),
                },
                User {
                    id: "admin".into(),
                    display_name: "R. Nair".into(),
                    roles: vec![Role::Employee],
                    department: Some("IT & Systems".into()),
                },
                User {
                    id: "kbadmin".into(),
                    display_name: "A. Fernandes".into(),
                    roles: vec![Role::Employee],
                    department: Some("Technical Services".into()),
                },
                User {
                    id: "engineer".into(),
                    display_name: "P. Shetty".into(),
                    roles: vec![Role::Employee],
                    department: Some("Inspection".into()),
                },
                User {
                    id: "reviewer".into(),
                    display_name: "M. Rao".into(),
                    roles: vec![Role::Employee],
                    department: Some("Maintenance".into()),
                },
                User {
                    id: "auditor".into(),
                    display_name: "V. Menon".into(),
                    roles: vec![Role::Employee],
                    department: Some("Internal Audit".into()),
                },
            ],
        }
    }

    pub fn all(&self) -> &[User] {
        &self.users
    }

    pub fn find(&self, id: &str) -> Option<&User> {
        self.users.iter().find(|u| u.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Permission::*;

    #[test]
    fn an_account_with_no_roles_holds_nothing() {
        let user = User::new("nobody", "New Account", vec![]);
        assert!(user.permissions().is_empty());
        for role in Role::ALL {
            let _ = role; // every permission is checked below via permissions()
        }
        assert!(!user.holds(UseModel));
        assert!(!user.holds(ViewAuditLog));
    }

    #[test]
    fn permissions_accumulate_across_roles() {
        let both = User::new("priya", "Priya", vec![Role::Administrator, Role::Employee]);
        assert!(both.holds(ModifyPolicy), "from Administrator");
        assert!(both.holds(UseModel), "from Employee");
        assert!(both.holds(EnterProvisioning), "from Administrator");
    }

    /// The Administrator is the superset of Employee: every Employee
    /// permission is held, plus the administrative ones. WriteFiles is
    /// intentionally not granted by any role — it is gated per-path by
    /// the policy gateway and always needs approval.
    #[test]
    fn the_administrator_holds_every_permission_except_writefiles() {
        let admin = User::new("root", "Administrator", vec![Role::Administrator]);
        for p in [
            UseModel,
            UploadDocument,
            SearchKnowledge,
            ExecuteCode,
            GenerateArtifact,
            ApproveOutput,
            ImportModel,
            ViewAuditLog,
            ModifyPolicy,
            EnterProvisioning,
        ] {
            assert!(admin.holds(p), "Administrator should hold {p:?}");
        }
        assert!(!admin.holds(WriteFiles));
    }

    /// Employee holds the everyday work permissions and not the administrative ones.
    #[test]
    fn the_employee_holds_only_work_permissions() {
        let employee = User::new("u", "Employee", vec![Role::Employee]);
        // Held.
        assert!(employee.holds(UseModel));
        assert!(employee.holds(UploadDocument));
        assert!(employee.holds(SearchKnowledge));
        assert!(employee.holds(ExecuteCode));
        assert!(employee.holds(GenerateArtifact));
        assert!(employee.holds(ApproveOutput));
        // Not held.
        assert!(!employee.holds(ImportModel));
        assert!(!employee.holds(ViewAuditLog));
        assert!(!employee.holds(ModifyPolicy));
        assert!(!employee.holds(EnterProvisioning));
        // WriteFiles is intentionally not granted by any role; it is
        // gated per-path by the policy gateway and always needs approval.
        assert!(!employee.holds(WriteFiles));
    }

    /// Legacy variants are not active roles and must not grant permissions.
    #[test]
    fn legacy_role_variants_grant_nothing() {
        for role in [
            Role::ModelAdministrator,
            Role::KnowledgeAdministrator,
            Role::User,
            Role::Reviewer,
            Role::Auditor,
        ] {
            for p in [
                UseModel,
                UploadDocument,
                SearchKnowledge,
                ExecuteCode,
                WriteFiles,
                GenerateArtifact,
                ApproveOutput,
                ImportModel,
                ViewAuditLog,
                ModifyPolicy,
                EnterProvisioning,
            ] {
                assert!(
                    !role.grants(p),
                    "legacy role {:?} unexpectedly grants {p:?}",
                    role
                );
            }
        }
    }

    #[test]
    fn the_seeded_directory_has_six_accounts_with_one_administrator() {
        let directory = UserDirectory::seeded();
        assert_eq!(directory.all().len(), 6);
        let admins: Vec<_> = directory
            .all()
            .iter()
            .filter(|u| u.roles.contains(&Role::Administrator))
            .map(|u| u.id.as_str())
            .collect();
        let employees: Vec<_> = directory
            .all()
            .iter()
            .filter(|u| u.roles.contains(&Role::Employee))
            .map(|u| u.id.as_str())
            .collect();
        // S. Kulkarni is the sole Administrator.
        assert_eq!(admins, vec!["modeladmin"]);
        // The other five are Employees. The ids preserve the historical
        // names so the live credential store and any saved references
        // continue to work.
        assert_eq!(
            employees,
            vec!["admin", "kbadmin", "engineer", "reviewer", "auditor"]
        );
        // Every account holds exactly one role.
        for u in directory.all() {
            assert_eq!(
                u.roles.len(),
                1,
                "{} should hold exactly one role, holds {:?}",
                u.id,
                u.roles
            );
        }
    }

    /// The administrator is the only one who may open the network.
    #[test]
    fn only_the_administrator_can_open_the_network() {
        let directory = UserDirectory::seeded();
        let openers: Vec<_> = directory
            .all()
            .iter()
            .filter(|u| u.holds(EnterProvisioning))
            .map(|u| u.id.as_str())
            .collect();
        assert_eq!(openers, vec!["modeladmin"]);
    }

    #[test]
    fn accounts_are_found_by_id_and_missing_ones_are_none() {
        let directory = UserDirectory::seeded();
        assert_eq!(directory.find("engineer").unwrap().display_name, "P. Shetty");
        assert!(directory.find("nobody").is_none());
    }

    #[test]
    fn headline_role_picks_administrator_when_held() {
        let admin = User::new("a", "A", vec![Role::Administrator]);
        assert_eq!(admin.headline_role(), Role::Administrator);
        let both = User::new("b", "B", vec![Role::Employee, Role::Administrator]);
        assert_eq!(both.headline_role(), Role::Administrator);
    }

    #[test]
    fn headline_role_falls_back_to_employee() {
        let empty = User::new("e", "E", vec![]);
        assert_eq!(empty.headline_role(), Role::Employee);
        let legacy_only = User::new("l", "L", vec![Role::Reviewer]);
        assert_eq!(legacy_only.headline_role(), Role::Employee);
    }
}
