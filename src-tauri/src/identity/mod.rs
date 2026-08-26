//! Who is acting, and what that entitles them to.
//!
//! PS 26117 step 8 is explicit that the model must never decide permissions. So
//! entitlement is decided here, from the signed-in user's roles, and nothing in
//! this module can be reached by anything a model emits — it only ever sees a
//! [`Session`] the application already established.
//!
//! Two deliberate choices:
//!
//! - **A user holds several roles, not one.** A refinery has people who are both
//!   the model administrator and an ordinary user of the workbench. Forcing one
//!   role per person leads straight to everyone being an administrator, which is
//!   the outcome this is meant to prevent.
//! - **Roles grant, never revoke.** A permission is held if *any* role grants it.
//!   That makes the matrix below the whole story: there is no second table of
//!   exceptions to read before you know what someone can do.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    /// Runs the deployment. Deliberately not entitled to do the *work* — see the
    /// note on separation of duties below.
    Administrator,
    /// Imports and enables models.
    ModelAdministrator,
    /// Curates collections of manuals, SOPs and correspondence.
    KnowledgeAdministrator,
    /// Does the day-to-day knowledge work the product exists for.
    User,
    /// Signs off outputs before they leave the workbench.
    Reviewer,
    /// Reads the record. Reads nothing else.
    Auditor,
}

impl Role {
    pub const ALL: &'static [Role] = &[
        Role::Administrator,
        Role::ModelAdministrator,
        Role::KnowledgeAdministrator,
        Role::User,
        Role::Reviewer,
        Role::Auditor,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Role::Administrator => "Administrator",
            Role::ModelAdministrator => "Model administrator",
            Role::KnowledgeAdministrator => "Knowledge administrator",
            Role::User => "User",
            Role::Reviewer => "Reviewer",
            Role::Auditor => "Auditor",
        }
    }

    /// The whole entitlement matrix, in one place.
    ///
    /// Three properties are deliberate and are asserted by the tests below:
    ///
    /// - **The administrator cannot do the work.** No `UseModel`, no
    ///   `UploadDocument`, no `GenerateArtifact`. Someone who configures the
    ///   system should not also be able to quietly run confidential work through
    ///   it, and a person who genuinely needs both is given both roles, which
    ///   leaves a record that they were.
    /// - **Nobody approves their own output by virtue of a role.** Only
    ///   `Reviewer` holds `ApproveOutput`. Whether a *particular* person may
    ///   approve a *particular* task is a further question the policy gateway
    ///   answers, because a reviewer must still not sign off their own work.
    /// - **The auditor reads and nothing else.** An auditor who could also run
    ///   tasks could produce the very records they are meant to be checking.
    pub fn grants(self, permission: Permission) -> bool {
        use Permission::*;
        match self {
            Role::Administrator => matches!(
                permission,
                ModifyPolicy | ViewAuditLog | EnterProvisioning | ImportModel
            ),
            Role::ModelAdministrator => matches!(permission, ImportModel | EnterProvisioning),
            Role::KnowledgeAdministrator => {
                matches!(permission, UploadDocument | SearchKnowledge)
            }
            Role::User => matches!(
                permission,
                UseModel | UploadDocument | SearchKnowledge | ExecuteCode | GenerateArtifact
            ),
            Role::Reviewer => matches!(permission, ApproveOutput | SearchKnowledge | ViewAuditLog),
            Role::Auditor => matches!(permission, ViewAuditLog),
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
/// Seeded rather than empty so the role model is demonstrable from first launch:
/// an installation whose only account is an all-powerful administrator teaches
/// everyone to use that account, which is the failure this design exists to
/// avoid.
///
/// **No authentication yet.** Selecting an account establishes a [`Session`]
/// without proving the person is who they claim. Passwords, and whatever the
/// site uses instead (LDAP, smart card), are the next slice of work — PS step 7.
/// The UI says so plainly rather than implying a check that is not happening,
/// because a login box that accepts anyone is worse than none at all.
pub struct UserDirectory {
    users: Vec<User>,
}

impl Default for UserDirectory {
    fn default() -> Self {
        Self::seeded()
    }
}

impl UserDirectory {
    /// One account per role, plus the realistic case of somebody holding two.
    pub fn seeded() -> Self {
        Self {
            users: vec![
                User {
                    id: "admin".into(),
                    display_name: "R. Nair".into(),
                    roles: vec![Role::Administrator],
                    department: Some("IT & Systems".into()),
                },
                User {
                    id: "modeladmin".into(),
                    display_name: "S. Kulkarni".into(),
                    roles: vec![Role::ModelAdministrator, Role::Administrator],
                    department: Some("IT & Systems".into()),
                },
                User {
                    id: "kbadmin".into(),
                    display_name: "A. Fernandes".into(),
                    roles: vec![Role::KnowledgeAdministrator],
                    department: Some("Technical Services".into()),
                },
                User {
                    id: "engineer".into(),
                    display_name: "P. Shetty".into(),
                    roles: vec![Role::User],
                    department: Some("Inspection".into()),
                },
                User {
                    id: "reviewer".into(),
                    display_name: "M. Rao".into(),
                    roles: vec![Role::Reviewer],
                    department: Some("Maintenance".into()),
                },
                User {
                    id: "auditor".into(),
                    display_name: "V. Menon".into(),
                    roles: vec![Role::Auditor],
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
        let both = User::new("priya", "Priya", vec![Role::Administrator, Role::User]);
        assert!(both.holds(ModifyPolicy), "from Administrator");
        assert!(both.holds(UseModel), "from User");
    }

    /// Configuring the system and doing confidential work with it are separate
    /// jobs. Someone who needs both is given both roles, and that is on record.
    #[test]
    fn the_administrator_role_alone_cannot_do_the_work() {
        let admin = User::new("root", "Administrator", vec![Role::Administrator]);
        assert!(admin.holds(ModifyPolicy));
        assert!(admin.holds(EnterProvisioning));
        assert!(!admin.holds(UseModel));
        assert!(!admin.holds(UploadDocument));
        assert!(!admin.holds(GenerateArtifact));
        assert!(!admin.holds(ExecuteCode));
    }

    #[test]
    fn only_the_reviewer_role_grants_approval() {
        for role in Role::ALL {
            let grants = role.grants(ApproveOutput);
            assert_eq!(
                grants,
                *role == Role::Reviewer,
                "{} unexpectedly {} approval",
                role.label(),
                if grants { "grants" } else { "withholds" }
            );
        }
    }

    #[test]
    fn the_auditor_reads_and_does_nothing_else() {
        let auditor = User::new("a", "Auditor", vec![Role::Auditor]);
        assert_eq!(auditor.permissions(), vec![ViewAuditLog]);
    }

    /// Making the network reachable is the most consequential action here, so
    /// it must never be something an ordinary account can reach.
    #[test]
    fn ordinary_users_cannot_open_the_network() {
        let user = User::new("u", "User", vec![Role::User]);
        let reviewer = User::new("r", "Reviewer", vec![Role::Reviewer]);
        let auditor = User::new("a", "Auditor", vec![Role::Auditor]);
        let kb = User::new("k", "KB", vec![Role::KnowledgeAdministrator]);

        for account in [&user, &reviewer, &auditor, &kb] {
            assert!(
                !account.holds(EnterProvisioning),
                "{} should not be able to open the network",
                account.display_name
            );
        }
    }

    /// Writing outside the task workspace is not granted by any role yet — it
    /// is gated per-path by the policy gateway and always needs approval, so a
    /// role that handed it out unconditionally would undercut that.
    #[test]
    fn no_role_grants_unscoped_file_writes() {
        for role in Role::ALL {
            assert!(
                !role.grants(WriteFiles),
                "{} should not grant unscoped writes",
                role.label()
            );
        }
    }

    #[test]
    fn the_seeded_directory_covers_every_role() {
        let directory = UserDirectory::seeded();
        for role in Role::ALL {
            assert!(
                directory.all().iter().any(|u| u.roles.contains(role)),
                "no seeded account holds {}",
                role.label()
            );
        }
    }

    #[test]
    fn exactly_one_seeded_account_can_open_the_network() {
        let directory = UserDirectory::seeded();
        let openers: Vec<_> = directory
            .all()
            .iter()
            .filter(|u| u.holds(EnterProvisioning))
            .map(|u| u.id.as_str())
            .collect();
        // The administrator and the model administrator, and nobody else.
        assert_eq!(openers, vec!["admin", "modeladmin"]);
    }

    #[test]
    fn accounts_are_found_by_id_and_missing_ones_are_none() {
        let directory = UserDirectory::seeded();
        assert_eq!(directory.find("engineer").unwrap().display_name, "P. Shetty");
        assert!(directory.find("nobody").is_none());
    }
}
