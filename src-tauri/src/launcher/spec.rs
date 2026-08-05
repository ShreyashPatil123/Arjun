//! Tool definitions for the Launch section.
//!
//! A tool is data, not code: name, how to check it is installed, how to install
//! it, how to start it, and which gateway endpoint it speaks. Sarathi ships a
//! verified set and merges in whatever the user adds, so a new tool does not
//! require a new release.
//!
//! Nothing here runs a process — this module is the vocabulary and the pure
//! decisions, so both are testable without touching the machine.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Which gateway endpoint a tool talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    /// `/v1/chat/completions` — opencode, openclaw, Cursor, Continue.
    OpenAi,
    /// `/v1/messages` — Claude Code.
    Anthropic,
}

/// Package managers Sarathi will delegate installation to.
///
/// Deliberately a closed set. Sarathi never downloads an installer itself; it
/// asks a manager the user already has, which verifies its own sources. A
/// user-supplied entry naming something else is rejected rather than executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PackageManager {
    Npm,
    Winget,
}

impl PackageManager {
    /// The executable to look for on PATH.
    pub fn program(&self) -> &'static str {
        match self {
            Self::Npm => "npm",
            Self::Winget => "winget",
        }
    }

    /// Arguments that install `package` globally, for the confirmation prompt
    /// and for the actual run — the user sees exactly what will execute.
    pub fn install_args(&self, package: &str) -> Vec<String> {
        match self {
            Self::Npm => vec!["install".into(), "-g".into(), package.into()],
            Self::Winget => vec![
                "install".into(),
                "--id".into(),
                package.into(),
                "-e".into(),
                "--accept-package-agreements".into(),
                "--accept-source-agreements".into(),
            ],
        }
    }

    /// The exact command line, for display before anything runs.
    pub fn command_line(&self, package: &str) -> String {
        format!("{} {}", self.program(), self.install_args(package).join(" "))
    }
}

/// How to tell whether a tool is really installed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectSpec {
    /// Executable name to look for.
    pub command: String,
    /// Argument that prints a version. Defaults to `--version`.
    #[serde(default = "default_version_arg")]
    pub version_arg: String,
    /// Text that must appear in the output, lowercased before comparison.
    ///
    /// This is what stops a name collision counting as an install: `continue`
    /// is a shell keyword, so a naive lookup "finds" it. Requiring the tool to
    /// identify itself rules that out.
    pub expect: String,
}

fn default_version_arg() -> String {
    "--version".to_string()
}

/// How to install a tool, when Sarathi can.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallSpec {
    pub manager: PackageManager,
    pub package: String,
}

/// How to start a tool, already connected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchSpec {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment for the child process. Values may contain the placeholders
    /// described on [`resolve_env`].
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// A complete tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    pub id: String,
    pub name: String,
    /// One line describing what it is, shown on the card.
    #[serde(default)]
    pub description: String,
    pub protocol: Protocol,
    pub detect: DetectSpec,
    /// Absent when Sarathi cannot install it — a GUI app, say. The card then
    /// links out instead of offering a button that would not work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install: Option<InstallSpec>,
    pub launch: LaunchSpec,
    /// True for entries the user added, so the UI can mark them unverified.
    #[serde(default)]
    pub user_defined: bool,
}

impl ToolSpec {
    /// Rejects entries that could not work, before the user meets a dead button.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("tool id must not be empty".into());
        }
        if self.name.trim().is_empty() {
            return Err(format!("tool '{}' has no name", self.id));
        }
        if self.detect.command.trim().is_empty() {
            return Err(format!("tool '{}' has no command to check for", self.id));
        }
        if self.detect.expect.trim().is_empty() {
            return Err(format!(
                "tool '{}' must say what its version output should contain, \
                 otherwise an unrelated program of the same name would pass",
                self.id
            ));
        }
        if self.launch.command.trim().is_empty() {
            return Err(format!("tool '{}' has no command to start", self.id));
        }
        Ok(())
    }
}

/// Placeholders available in [`LaunchSpec::env`] values.
///
/// The gateway address is substituted at launch, never stored: the port is
/// user-configurable, so a saved address goes stale the moment it changes.
pub const PLACEHOLDER_BASE: &str = "{gatewayUrl}";
pub const PLACEHOLDER_BASE_V1: &str = "{gatewayUrlV1}";

/// Fills placeholders with the address the gateway is actually listening on.
pub fn resolve_env(env: &HashMap<String, String>, port: u16) -> HashMap<String, String> {
    let base = format!("http://127.0.0.1:{port}");
    let v1 = format!("{base}/v1");

    env.iter()
        .map(|(k, v)| {
            let filled = v
                .replace(PLACEHOLDER_BASE_V1, &v1)
                .replace(PLACEHOLDER_BASE, &base);
            (k.clone(), filled)
        })
        .collect()
}

/// Decides whether version output proves this is the expected tool.
///
/// Split out from process handling so the rule itself is testable.
pub fn output_identifies_tool(output: &str, expect: &str) -> bool {
    let needle = expect.trim().to_lowercase();
    !needle.is_empty() && output.to_lowercase().contains(&needle)
}

/// Tools Sarathi ships with, verified against their real install and launch
/// commands. Users can override any of these by id, or add their own.
pub fn builtin_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            id: "claude-code".into(),
            name: "Claude Code".into(),
            description: "Anthropic's coding agent for the terminal.".into(),
            protocol: Protocol::Anthropic,
            detect: DetectSpec {
                command: "claude".into(),
                version_arg: "--version".into(),
                expect: "claude".into(),
            },
            install: Some(InstallSpec {
                manager: PackageManager::Npm,
                package: "@anthropic-ai/claude-code".into(),
            }),
            launch: LaunchSpec {
                command: "claude".into(),
                args: vec![],
                env: HashMap::from([
                    ("ANTHROPIC_BASE_URL".to_string(), PLACEHOLDER_BASE.to_string()),
                    // Claude Code expects a key to be present; the gateway does
                    // not check it, but an empty value makes the client error
                    // before it ever sends a request.
                    ("ANTHROPIC_API_KEY".to_string(), "sarathi-local".to_string()),
                ]),
            },
            user_defined: false,
        },
        ToolSpec {
            id: "opencode".into(),
            name: "opencode".into(),
            description: "Open-source terminal coding agent.".into(),
            protocol: Protocol::OpenAi,
            detect: DetectSpec {
                command: "opencode".into(),
                version_arg: "--version".into(),
                expect: "opencode".into(),
            },
            install: Some(InstallSpec {
                manager: PackageManager::Npm,
                package: "opencode-ai".into(),
            }),
            launch: LaunchSpec {
                command: "opencode".into(),
                args: vec![],
                env: HashMap::from([
                    ("OPENAI_BASE_URL".to_string(), PLACEHOLDER_BASE_V1.to_string()),
                    ("OPENAI_API_KEY".to_string(), "sarathi-local".to_string()),
                ]),
            },
            user_defined: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_filled_with_the_live_port() {
        let env = HashMap::from([
            ("ANTHROPIC_BASE_URL".to_string(), PLACEHOLDER_BASE.to_string()),
            ("OPENAI_BASE_URL".to_string(), PLACEHOLDER_BASE_V1.to_string()),
            ("OPENAI_API_KEY".to_string(), "sarathi-local".to_string()),
        ]);

        let resolved = resolve_env(&env, 11435);

        assert_eq!(resolved["ANTHROPIC_BASE_URL"], "http://127.0.0.1:11435");
        assert_eq!(resolved["OPENAI_BASE_URL"], "http://127.0.0.1:11435/v1");
        assert_eq!(resolved["OPENAI_API_KEY"], "sarathi-local", "plain values pass through");
    }

    #[test]
    fn a_changed_port_changes_the_address() {
        // The address must never be cached: the port is user-configurable.
        let env = HashMap::from([("BASE".to_string(), PLACEHOLDER_BASE.to_string())]);

        assert_eq!(resolve_env(&env, 11435)["BASE"], "http://127.0.0.1:11435");
        assert_eq!(resolve_env(&env, 9999)["BASE"], "http://127.0.0.1:9999");
    }

    #[test]
    fn the_v1_placeholder_is_not_mangled_by_the_shorter_one() {
        // `{gatewayUrl}` is a prefix of `{gatewayUrlV1}`; replacing in the wrong
        // order would leave a stray "V1" on the end of the address.
        let env = HashMap::from([("B".to_string(), PLACEHOLDER_BASE_V1.to_string())]);

        assert_eq!(resolve_env(&env, 11435)["B"], "http://127.0.0.1:11435/v1");
    }

    #[test]
    fn version_output_must_name_the_tool() {
        assert!(output_identifies_tool("claude 1.2.3", "claude"));
        assert!(output_identifies_tool("OpenCode v0.4", "opencode"), "match is case-insensitive");

        // The regression this rule exists for: `continue` is a shell keyword,
        // so a naive lookup reports it as installed.
        assert!(!output_identifies_tool("", "continue"));
        assert!(!output_identifies_tool("bash: continue: usage", "opencode"));
    }

    #[test]
    fn an_empty_expectation_never_matches() {
        // Otherwise every program would pass the check.
        assert!(!output_identifies_tool("anything at all", ""));
        assert!(!output_identifies_tool("anything at all", "   "));
    }

    #[test]
    fn install_commands_are_shown_exactly_as_they_will_run() {
        let npm = PackageManager::Npm;
        assert_eq!(npm.command_line("opencode-ai"), "npm install -g opencode-ai");
        assert_eq!(npm.install_args("x"), vec!["install", "-g", "x"]);

        let winget = PackageManager::Winget;
        assert!(winget.command_line("Some.Id").starts_with("winget install --id Some.Id -e"));
    }

    #[test]
    fn builtin_tools_are_all_valid() {
        let tools = builtin_tools();
        assert!(!tools.is_empty());

        for t in &tools {
            t.validate().unwrap_or_else(|e| panic!("shipped tool is invalid: {e}"));
            assert!(!t.user_defined, "shipped tools are not user-defined");
        }
    }

    #[test]
    fn builtin_ids_are_unique() {
        let tools = builtin_tools();
        let mut ids: Vec<&str> = tools.iter().map(|t| t.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate tool ids would make overrides ambiguous");
    }

    #[test]
    fn each_builtin_points_at_the_endpoint_it_speaks() {
        for t in builtin_tools() {
            let keys: Vec<&str> = t.launch.env.keys().map(String::as_str).collect();
            match t.protocol {
                Protocol::Anthropic => assert!(
                    keys.contains(&"ANTHROPIC_BASE_URL"),
                    "{} speaks Anthropic but never receives that address",
                    t.id
                ),
                Protocol::OpenAi => assert!(
                    keys.contains(&"OPENAI_BASE_URL"),
                    "{} speaks OpenAI but never receives that address",
                    t.id
                ),
            }
        }
    }

    #[test]
    fn anthropic_tools_get_the_base_url_and_openai_tools_get_v1() {
        // The OpenAI endpoint lives under /v1; the Anthropic one does not.
        for t in builtin_tools() {
            match t.protocol {
                Protocol::Anthropic => {
                    assert_eq!(t.launch.env["ANTHROPIC_BASE_URL"], PLACEHOLDER_BASE)
                }
                Protocol::OpenAi => {
                    assert_eq!(t.launch.env["OPENAI_BASE_URL"], PLACEHOLDER_BASE_V1)
                }
            }
        }
    }

    fn valid_spec() -> ToolSpec {
        builtin_tools().into_iter().next().unwrap()
    }

    #[test]
    fn an_entry_without_an_expectation_is_rejected() {
        // Without it, any program of the same name would count as installed.
        let mut spec = valid_spec();
        spec.detect.expect = "  ".into();

        let err = spec.validate().unwrap_err();
        assert!(err.contains("version output"), "error should explain why: {err}");
    }

    #[test]
    fn entries_missing_essentials_are_rejected() {
        for mutate in [
            (|s: &mut ToolSpec| s.id = "".into()) as fn(&mut ToolSpec),
            |s: &mut ToolSpec| s.name = "".into(),
            |s: &mut ToolSpec| s.detect.command = "".into(),
            |s: &mut ToolSpec| s.launch.command = "".into(),
        ] {
            let mut spec = valid_spec();
            mutate(&mut spec);
            assert!(spec.validate().is_err(), "invalid entry should not pass validation");
        }
    }

    #[test]
    fn a_tool_entry_survives_a_round_trip_through_json() {
        // User entries are read from a file, so the format must be stable.
        let original = valid_spec();
        let json = serde_json::to_string(&original).expect("serialise");
        let parsed: ToolSpec = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed.id, original.id);
        assert_eq!(parsed.protocol, original.protocol);
        assert_eq!(parsed.launch.env, original.launch.env);
    }

    #[test]
    fn version_arg_defaults_when_a_user_entry_omits_it() {
        let json = r#"{
            "id": "mytool",
            "name": "My Tool",
            "protocol": "openai",
            "detect": { "command": "mytool", "expect": "mytool" },
            "launch": { "command": "mytool" }
        }"#;

        let spec: ToolSpec = serde_json::from_str(json).expect("minimal entry should parse");

        assert_eq!(spec.detect.version_arg, "--version");
        assert!(spec.install.is_none(), "install is optional");
        assert!(spec.launch.args.is_empty());
        assert!(spec.validate().is_ok());
    }
}
