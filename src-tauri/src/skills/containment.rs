//! Resolving a reference, script or asset a skill names — inside the skill.
//!
//! A `SKILL.md` says things like `see references/seal-limits.md`. That string
//! was written by whoever wrote the skill, and a skill is not trusted, so it is
//! the same class of input as a path a model emitted: it may be a traversal, an
//! absolute path, a Windows drive letter, or a name that means something else
//! on a case-insensitive filesystem.
//!
//! The rule is the one the tool gateway already uses, applied to a second kind
//! of caller: **resolve `..` textually, then confirm the result is under the
//! skill's own directory.** Textual because the file may legitimately not exist
//! (a skill can reference something an operator has not installed), and because
//! a textual check cannot be defeated by a link planted between the check and
//! the read.
//!
//! Only the three declared subdirectories are reachable. A skill cannot read
//! its own `SKILL.md` back through this path, and cannot reach a sibling skill
//! — which matters, because sibling skills include ones an operator has
//! deliberately quarantined.

use std::path::{Component, Path, PathBuf};

/// The subdirectories a skill may name.
///
/// A closed list rather than "anything under the skill root". The format has
/// three folders; permitting a fourth by accident is how a skill ends up
/// reading a `.git` directory or a stray export somebody left beside it.
pub const REACHABLE: &[&str] = &["references", "scripts", "assets"];

/// Why a named path was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// It is absolute, or carries a drive or share root.
    Rooted,
    /// It climbs out of the skill directory.
    Escapes,
    /// Its first segment is not one of [`REACHABLE`].
    NotReachable { segment: String },
    /// It names nothing.
    Empty,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Rooted => write!(
                f,
                "a skill may only name paths relative to its own directory, and this one has a root"
            ),
            Refusal::Escapes => write!(
                f,
                "this path climbs out of the skill's own directory, which is not permitted"
            ),
            Refusal::NotReachable { segment } => write!(
                f,
                "a skill may only read from references/, scripts/ and assets/, and this names {segment:?}"
            ),
            Refusal::Empty => write!(f, "this path names nothing"),
        }
    }
}

/// Resolves a path a skill named, against the skill's own root.
///
/// Returns the absolute path only when it stays inside. The file is not opened
/// and need not exist — this answers "may it be read", and the caller answers
/// "is it there".
pub fn resolve(skill_root: &Path, named: &str) -> Result<PathBuf, Refusal> {
    let named = named.trim();
    if named.is_empty() {
        return Err(Refusal::Empty);
    }

    let candidate = Path::new(named);
    // A rooted path is refused rather than joined. `Path::join` *replaces* when
    // its argument has a root, so joining would silently produce a path outside
    // the skill and leave a later check to catch it.
    if candidate.is_absolute() || candidate.has_root() {
        return Err(Refusal::Rooted);
    }
    if candidate.components().any(|c| matches!(c, Component::Prefix(_))) {
        return Err(Refusal::Rooted);
    }

    let mut relative = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // Refused rather than clamped: clamping turns a traversal into
                // a valid read of somewhere unexpected.
                if !relative.pop() {
                    return Err(Refusal::Escapes);
                }
            }
            Component::Normal(part) => relative.push(part),
            Component::RootDir | Component::Prefix(_) => return Err(Refusal::Rooted),
        }
    }

    let mut segments = relative.components();
    let first = segments
        .next()
        .and_then(|c| match c {
            Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .ok_or(Refusal::Empty)?;

    // Compared case-insensitively, because a skill naming `References/` on a
    // case-insensitive filesystem would otherwise pass the reachability check
    // by not matching, and then resolve to the same directory anyway.
    if !REACHABLE
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(first))
    {
        return Err(Refusal::NotReachable {
            segment: first.to_string(),
        });
    }
    // A bare `references` with nothing after it is a directory, not a file.
    if segments.next().is_none() {
        return Err(Refusal::Empty);
    }

    Ok(skill_root.join(relative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        PathBuf::from("/skills/inspection-approval-note")
    }

    #[test]
    fn an_ordinary_reference_resolves_under_the_skill() {
        let resolved = resolve(&root(), "references/seal-limits.md").expect("resolves");
        assert!(resolved.starts_with(root()));
        assert!(resolved.ends_with("seal-limits.md"));
    }

    #[test]
    fn all_three_declared_directories_are_reachable() {
        for folder in REACHABLE {
            assert!(resolve(&root(), &format!("{folder}/thing.txt")).is_ok(), "{folder}");
        }
    }

    #[test]
    fn a_traversal_is_refused_rather_than_clamped() {
        // Clamping would turn this into a valid read of a sibling skill —
        // including one an operator deliberately quarantined.
        assert_eq!(
            resolve(&root(), "references/../../other-skill/SKILL.md"),
            Err(Refusal::Escapes)
        );
        assert_eq!(resolve(&root(), "../secrets.md"), Err(Refusal::Escapes));
    }

    #[test]
    fn an_absolute_path_is_refused_before_it_is_joined() {
        // `Path::join` replaces rather than appends when the argument has a
        // root, so joining first and checking after would produce a path
        // outside the skill and rely on the later check to notice.
        assert_eq!(resolve(&root(), "/etc/passwd"), Err(Refusal::Rooted));
        assert_eq!(resolve(&root(), "C:/Windows/System32"), Err(Refusal::Rooted));
        assert_eq!(resolve(&root(), r"\\server\share\x"), Err(Refusal::Rooted));
    }

    #[test]
    fn the_skill_cannot_read_its_own_definition_back() {
        // Nothing good comes of it, and a skill that could would be a skill
        // that could quote its own frontmatter at the model as though it were
        // policy.
        assert_eq!(
            resolve(&root(), "SKILL.md"),
            Err(Refusal::NotReachable {
                segment: "SKILL.md".to_string()
            })
        );
    }

    #[test]
    fn a_directory_outside_the_three_is_refused_by_name() {
        assert_eq!(
            resolve(&root(), ".git/config"),
            Err(Refusal::NotReachable {
                segment: ".git".to_string()
            })
        );
    }

    #[test]
    fn a_bare_directory_is_not_a_file() {
        assert_eq!(resolve(&root(), "references"), Err(Refusal::Empty));
        assert_eq!(resolve(&root(), "references/"), Err(Refusal::Empty));
    }

    #[test]
    fn an_empty_name_is_refused() {
        assert_eq!(resolve(&root(), "   "), Err(Refusal::Empty));
    }

    #[test]
    fn a_differently_cased_directory_does_not_slip_past_the_check() {
        // On a case-insensitive filesystem `References/` reaches the same
        // folder, so treating it as unreachable-and-therefore-refused would be
        // right by accident and wrong in principle. It is reachable, and named.
        assert!(resolve(&root(), "References/x.md").is_ok());
    }
}
