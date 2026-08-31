//! Role-based access control isolation tests.
//!
//! These tests verify the 2-role permission matrix against the real
//! `Role::grants(permission)` function the back-end uses on every gated
//! Tauri command. The IPC layer is a thin shim around this function
//! (`require_permission` in `commands/governance.rs`), so testing the
//! matrix is testing the boundary.
//!
//! ## Why this is not a black-box IPC test
//!
//! Each Tauri command takes `State<'_, T>` arguments that the
//! `tauri` crate's test utilities can construct, but a black-box
//! call would also need a `tauri::App` for `AppHandle` and the
//! `tauri::test::mock_app` requires the `test` feature, which the
//! production crate does not enable. The matrix test below proves
//! the same property without that scaffolding, and is the test the
//! CI pipeline can run.
//!
//! ## The matrix, in one place
//!
//! The test data below is the human-readable version of
//! `Role::grants(permission)` in `src-tauri/src/identity/mod.rs`. The
//! two must agree. The test fails on disagreement, so a drift is
//! caught immediately.

use sarathi_lib::identity::{Permission, Role};

/// Each row is one permission; the two active roles are the only
/// columns that matter — Administrator and Employee. Legacy
/// variants (`ModelAdministrator`, `KnowledgeAdministrator`,
/// `User`, `Reviewer`, `Auditor`) are kept in the enum for test
/// compatibility, but the matrix test asserts they grant
/// nothing.
const MATRIX: &[(Permission, bool, bool, bool, bool, bool, bool, bool)] = &[
    // (Permission, Admin, Employee, ModelAdmin, KBAdmin, User, Reviewer, Auditor)
    // The legacy columns are pinned to `false` — those roles do not
    // exist in the active product. We keep them here so a regression
    // that re-enables them is caught immediately.
    (Permission::UseModel,           true,  true,  false, false, false, false, false),
    (Permission::UploadDocument,     true,  true,  false, false, false, false, false),
    (Permission::SearchKnowledge,    true,  true,  false, false, false, false, false),
    (Permission::ExecuteCode,        true,  true,  false, false, false, false, false),
    (Permission::WriteFiles,         false, false, false, false, false, false, false),
    (Permission::GenerateArtifact,   true,  true,  false, false, false, false, false),
    (Permission::ApproveOutput,      true,  true,  false, false, false, false, false),
    (Permission::ImportModel,        true,  false, false, false, false, false, false),
    (Permission::ViewAuditLog,       true,  false, false, false, false, false, false),
    (Permission::ModifyPolicy,       true,  false, false, false, false, false, false),
    (Permission::EnterProvisioning,  true,  false, false, false, false, false, false),
];

/// The matrix and the implementation must agree. A drift in either
/// direction is a security incident: granting a new permission is
/// expansion of authority; removing one is a silent refusal.
#[test]
fn role_grants_matches_documented_matrix() {
    for (permission, admin, employee, modeladmin, kbadmin, user, reviewer, auditor) in MATRIX {
        assert_eq!(Role::Administrator.grants(*permission), *admin, "Administrator / {permission:?}");
        assert_eq!(Role::Employee.grants(*permission), *employee, "Employee / {permission:?}");
        // Legacy variants grant nothing in the active product.
        assert_eq!(Role::ModelAdministrator.grants(*permission), *modeladmin, "ModelAdmin / {permission:?}");
        assert_eq!(Role::KnowledgeAdministrator.grants(*permission), *kbadmin, "KBAdmin / {permission:?}");
        assert_eq!(Role::User.grants(*permission), *user, "User / {permission:?}");
        assert_eq!(Role::Reviewer.grants(*permission), *reviewer, "Reviewer / {permission:?}");
        assert_eq!(Role::Auditor.grants(*permission), *auditor, "Auditor / {permission:?}");
    }
}

/// A user with no roles holds nothing. An account that exists but
/// has been given no job cannot do anything privileged.
#[test]
fn a_user_with_no_roles_holds_no_permissions() {
    let user = sarathi_lib::identity::User::new("nobody", "Nobody", vec![]);
    for permission in [
        Permission::UseModel,
        Permission::ImportModel,
        Permission::ViewAuditLog,
        Permission::ModifyPolicy,
    ] {
        assert!(!user.holds(permission), "empty-role user holds {permission:?}");
    }
}

/// A user with two roles holds the union of what each grants. A
/// person who needs both kinds of work is given both roles, and the
/// audit chain names them both.
#[test]
fn a_user_with_two_roles_holds_the_union_of_what_each_grants() {
    let user = sarathi_lib::identity::User::new(
        "root",
        "Administrator",
        vec![Role::Administrator, Role::Employee],
    );
    // Administrator is the superset: every permission.
    assert!(user.holds(Permission::ApproveOutput));
    assert!(user.holds(Permission::ImportModel));
    assert!(user.holds(Permission::ModifyPolicy));
    assert!(user.holds(Permission::EnterProvisioning));
    assert!(user.holds(Permission::ViewAuditLog));
    assert!(user.holds(Permission::UseModel));
    assert!(user.holds(Permission::UploadDocument));
    assert!(user.holds(Permission::SearchKnowledge));
    assert!(user.holds(Permission::ExecuteCode));
    assert!(user.holds(Permission::GenerateArtifact));
}

/// The superset property: Administrator holds every permission an
/// Employee holds, and then more. This is the whole point of
/// collapsing the six-role matrix into a 2-role one.
#[test]
fn administrator_is_a_superset_of_employee() {
    let admin = sarathi_lib::identity::User::new("a", "A", vec![Role::Administrator]);
    let employee = sarathi_lib::identity::User::new("e", "E", vec![Role::Employee]);
    for permission in [
        Permission::UseModel,
        Permission::UploadDocument,
        Permission::SearchKnowledge,
        Permission::ExecuteCode,
        Permission::GenerateArtifact,
        Permission::ApproveOutput,
        Permission::ImportModel,
        Permission::ViewAuditLog,
        Permission::ModifyPolicy,
        Permission::EnterProvisioning,
    ] {
        if employee.holds(permission) {
            assert!(
                admin.holds(permission),
                "Administrator should be a superset of Employee, but does not hold {permission:?}"
            );
        }
    }
}

/// Employee cannot perform any administrative action. This is the
/// negative half of the superset property.
#[test]
fn employee_cannot_perform_administrative_actions() {
    let employee = sarathi_lib::identity::User::new("e", "E", vec![Role::Employee]);
    for permission in [
        Permission::ImportModel,
        Permission::ViewAuditLog,
        Permission::ModifyPolicy,
        Permission::EnterProvisioning,
    ] {
        assert!(
            !employee.holds(permission),
            "Employee should not be able to {permission:?}"
        );
    }
}

/// The seeded directory exposes exactly the two active roles.
#[test]
fn the_seeded_directory_exposes_only_active_roles() {
    use sarathi_lib::identity::UserDirectory;
    let directory = UserDirectory::seeded();
    let all_held: std::collections::HashSet<Role> =
        directory.all().iter().flat_map(|u| u.roles.iter().copied()).collect();
    for role in all_held {
        assert!(role.is_active(), "seeded directory exposes inactive role {role:?}");
    }
    // Exactly one Administrator: S. Kulkarni. The other five seeded
    // accounts are Employees.
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
    assert_eq!(admins, vec!["modeladmin"]);
    assert_eq!(
        employees,
        vec!["admin", "kbadmin", "engineer", "reviewer", "auditor"]
    );
}

/// `User::headline_role` collapses multi-role accounts to the single
/// "what the menu should call me" role.
#[test]
fn headline_role_uses_active_roles_only() {
    let u = sarathi_lib::identity::User::new("x", "X", vec![Role::Administrator, Role::Employee]);
    assert_eq!(u.headline_role(), Role::Administrator);
    let u2 = sarathi_lib::identity::User::new("y", "Y", vec![Role::Employee]);
    assert_eq!(u2.headline_role(), Role::Employee);
    let u3 = sarathi_lib::identity::User::new("z", "Z", vec![]);
    assert_eq!(u3.headline_role(), Role::Employee);
}

/// Pinning: every seeded account holds exactly one role. The
/// 2-role contract is "one role per account", so the headline
/// role is unambiguous and the menu can render a single label.
/// A regression that grants two roles to one seeded account is
/// caught here, before the menu renders the wrong title.
#[test]
fn every_seeded_account_holds_exactly_one_role() {
    use sarathi_lib::identity::UserDirectory;
    let directory = UserDirectory::seeded();
    for user in directory.all() {
        assert_eq!(
            user.roles.len(),
            1,
            "{} ({}) holds {} roles; expected exactly one",
            user.display_name,
            user.id,
            user.roles.len()
        );
    }
}

/// Pinning: `User::headline_role` always returns an active role,
/// regardless of what legacy variants the user happens to hold.
/// The front-end renders the result directly in the account menu,
/// so a legacy role string must not leak into the UI even if the
/// directory contains one.
#[test]
fn headline_role_never_returns_a_legacy_role() {
    use sarathi_lib::identity::{Role, User};
    let legacy = User::new("x", "X", vec![Role::ModelAdministrator, Role::Auditor]);
    assert!(legacy.headline_role().is_active());
    let empty = User::new("y", "Y", vec![]);
    assert!(empty.headline_role().is_active());
}

/// Pinning: an Employee who calls `require_permission` with
/// `ModifyPolicy` is refused at the exact chokepoint the
/// back-end uses — not at the matrix, not at the policy, but
/// at `require_permission`. This is the test that catches a
/// regression that lets an Employee call set_config,
/// set_hf_token, or override_hardware_value.
#[test]
fn require_permission_refuses_employee_for_modify_policy() {
    use sarathi_lib::commands::governance::{require_permission, CurrentSession};
    use sarathi_lib::identity::{Permission, Session, User};
    use std::sync::Arc;
    use std::sync::RwLock;

    let employee_user = User::new("engineer", "P. Shetty", vec![Role::Employee]);
    let session = Session::open(employee_user);
    let current: CurrentSession = Arc::new(RwLock::new(Some(session)));

    let result = require_permission(&current, Permission::ModifyPolicy);
    assert!(
        result.is_err(),
        "Employee must be refused ModifyPolicy at the require_permission chokepoint"
    );
    let message = result.unwrap_err();
    assert!(
        message.contains("not permitted"),
        "refusal message should explain why, got: {message}"
    );
}

/// Pinning: the same Employee is admitted for the work-class
/// permissions, proving the negative test above is not just an
/// "always refuse" bug.
#[test]
fn require_permission_admits_employee_for_use_model() {
    use sarathi_lib::commands::governance::{require_permission, CurrentSession};
    use sarathi_lib::identity::{Permission, Session, User};
    use std::sync::Arc;
    use std::sync::RwLock;

    let employee_user = User::new("engineer", "P. Shetty", vec![Role::Employee]);
    let session = Session::open(employee_user);
    let current: CurrentSession = Arc::new(RwLock::new(Some(session)));

    let result = require_permission(&current, Permission::UseModel);
    assert!(
        result.is_ok(),
        "Employee should be admitted UseModel, got: {:?}",
        result.err()
    );
}

/// Pinning: an unauthenticated caller is refused at the
/// `require_permission` chokepoint. This is the test that catches
/// a regression that lets a request through without a signed-in
/// user.
#[test]
fn require_permission_refuses_an_unauthenticated_caller() {
    use sarathi_lib::commands::governance::{require_permission, CurrentSession};
    use sarathi_lib::identity::Permission;
    use std::sync::Arc;
    use std::sync::RwLock;

    let current: CurrentSession = Arc::new(RwLock::new(None));
    let result = require_permission(&current, Permission::UseModel);
    assert!(
        result.is_err(),
        "Unauthenticated caller must be refused, even for the most basic permission"
    );
}
