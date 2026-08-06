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

/// A config file Sarathi writes for the tool before starting it.
///
/// Tools that keep their own provider settings will use them unless pointed at
/// a different file. Generating one per launch is what makes the tool a client
/// of Sarathi rather than of whatever it was configured with previously.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientConfig {
    /// File name, written inside the tool's isolated client directory.
    pub file_name: String,
    /// Contents, with the same placeholders as [`LaunchSpec::env`]. Values are
    /// JSON-escaped on substitution, so a Windows path cannot break the file.
    pub contents: String,
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
    /// Inherited variables to strip before starting the tool.
    ///
    /// Sarathi cannot win by only *setting* variables: a tool that reads
    /// `OPENAI_API_KEY` or `ANTHROPIC_AUTH_TOKEN` from the user's environment
    /// will authenticate against that provider no matter what else is set.
    /// These are removed so the child sees only what Sarathi gives it.
    #[serde(default)]
    pub env_remove: Vec<String>,
    /// Config file to generate before launching, if the tool needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_config: Option<ClientConfig>,
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
/// The loaded model, in a form safe to use as a client-side model id.
pub const PLACEHOLDER_MODEL: &str = "{model}";
/// The loaded model's display name.
pub const PLACEHOLDER_MODEL_NAME: &str = "{modelName}";
/// This tool's isolated config directory.
pub const PLACEHOLDER_CLIENT_DIR: &str = "{clientDir}";

/// Everything Sarathi knows at the moment a tool is started.
///
/// Passed as one value so a launch cannot be assembled from a stale port and a
/// fresh model, or vice versa.
#[derive(Debug, Clone)]
pub struct LaunchContext {
    pub port: u16,
    /// Model id as the gateway reports it, e.g. `Qwen/Qwen2.5-3B`.
    pub model_id: String,
    /// Human-facing name, e.g. `Qwen2.5-3B`.
    pub model_name: String,
    /// Directory this tool's generated config lives in.
    pub client_dir: String,
}

impl LaunchContext {
    /// A model id a client can use without ambiguity.
    ///
    /// opencode addresses models as `provider/model`, so a real id containing a
    /// slash would be split in the wrong place. Sarathi's gateway ignores the
    /// requested model and always serves the loaded one, so a sanitized id
    /// costs nothing and routes identically.
    pub fn client_model_id(&self) -> String {
        let cleaned: String = self
            .model_id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '-' })
            .collect();

        let trimmed = cleaned.trim_matches('-').to_string();
        if trimmed.is_empty() { "sarathi-model".to_string() } else { trimmed }
    }
}

/// Fills placeholders with the live gateway address and loaded model.
pub fn resolve_env(env: &HashMap<String, String>, ctx: &LaunchContext) -> HashMap<String, String> {
    env.iter()
        .map(|(k, v)| (k.clone(), fill_placeholders(v, ctx, false)))
        .collect()
}

/// Substitutes placeholders in `template`.
///
/// `json` escapes the substituted values, because generated config files are
/// JSON and a Windows client directory is full of backslashes.
pub fn fill_placeholders(template: &str, ctx: &LaunchContext, json: bool) -> String {
    let base = format!("http://127.0.0.1:{}", ctx.port);
    let v1 = format!("{base}/v1");

    let esc = |value: &str| -> String {
        if json {
            value.replace('\\', "\\\\").replace('"', "\\\"")
        } else {
            value.to_string()
        }
    };

    // The v1 placeholder is replaced first: the shorter one is a prefix of it,
    // and the other order would leave a stray "V1" on the end of the address.
    template
        .replace(PLACEHOLDER_BASE_V1, &esc(&v1))
        .replace(PLACEHOLDER_BASE, &esc(&base))
        .replace(PLACEHOLDER_MODEL_NAME, &esc(&ctx.model_name))
        .replace(PLACEHOLDER_MODEL, &esc(&ctx.client_model_id()))
        .replace(PLACEHOLDER_CLIENT_DIR, &esc(&ctx.client_dir))
}

/// Decides whether version output proves this is the expected tool.
///
/// Split out from process handling so the rule itself is testable.
pub fn output_identifies_tool(output: &str, expect: &str) -> bool {
    let needle = expect.trim().to_lowercase();
    if needle.is_empty() {
        return false;
    }
    if output.to_lowercase().contains(&needle) {
        return true;
    }

    // Some tools answer `--version` with nothing but the number: opencode
    // prints "1.18.13". Demanding its own name back left a genuinely installed
    // tool showing Install forever, and re-installing never changed that.
    // A lone version is still evidence — the cases this rule guards against
    // (a shell keyword, an error message) do not produce one.
    is_bare_version(output)
}

/// True when the first non-empty line is only a dotted version number.
fn is_bare_version(output: &str) -> bool {
    let Some(line) = output.lines().map(str::trim).find(|l| !l.is_empty()) else {
        return false;
    };

    let digits = line.strip_prefix('v').unwrap_or(line);
    let parts: Vec<&str> = digits.split('.').collect();

    parts.len() >= 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
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
                    // Exactly one credential is set. Claude Code warns when it
                    // finds both a token and a key, and picks between them by
                    // its own precedence — which is how a user's real account
                    // kept winning over Sarathi. The gateway ignores the value.
                    ("ANTHROPIC_AUTH_TOKEN".to_string(), "sarathi-local".to_string()),
                    // Without these it defaults to a Claude model name and
                    // reports that in its status line, even though every
                    // request is answered by whatever Sarathi has loaded.
                    ("ANTHROPIC_MODEL".to_string(), PLACEHOLDER_MODEL.to_string()),
                    ("ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(), PLACEHOLDER_MODEL.to_string()),
                    ("ANTHROPIC_SMALL_FAST_MODEL".to_string(), PLACEHOLDER_MODEL.to_string()),
                    // A private config directory. The logged-in account lives in
                    // the default one, so pointing elsewhere is what stops the
                    // tool resuming the user's own Anthropic session.
                    ("CLAUDE_CONFIG_DIR".to_string(), PLACEHOLDER_CLIENT_DIR.to_string()),
                    ("DISABLE_AUTOUPDATER".to_string(), "1".to_string()),
                    ("CLAUDE_CODE_ENABLE_TELEMETRY".to_string(), "0".to_string()),
                ]),
                env_remove: vec![
                    // Any of these would re-point the tool at a real provider.
                    "ANTHROPIC_API_KEY".into(),
                    "ANTHROPIC_CUSTOM_HEADERS".into(),
                    "CLAUDE_CODE_OAUTH_TOKEN".into(),
                    "CLAUDE_CODE_USE_BEDROCK".into(),
                    "CLAUDE_CODE_USE_VERTEX".into(),
                    // Session identity of a parent Claude Code, if Sarathi was
                    // itself started from one. Inheriting it confuses the child.
                    "CLAUDECODE".into(),
                    "CLAUDE_CODE_ENTRYPOINT".into(),
                    "CLAUDE_CODE_SESSION_ID".into(),
                    "CLAUDE_CODE_HOST_SESSION_ID".into(),
                    "CLAUDE_CODE_CHILD_SESSION".into(),
                ],
                client_config: None,
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
                    // Points at the config Sarathi generates below, so the
                    // user's own opencode.json is never consulted.
                    (
                        "OPENCODE_CONFIG".to_string(),
                        format!("{PLACEHOLDER_CLIENT_DIR}/opencode.json"),
                    ),
                ]),
                env_remove: vec![
                    // opencode discovers providers from whichever API keys it
                    // finds in the environment. Left in place, these are why it
                    // opened on an OpenAI model instead of Sarathi's.
                    "OPENAI_API_KEY".into(),
                    "OPENAI_BASE_URL".into(),
                    "ANTHROPIC_API_KEY".into(),
                    "ANTHROPIC_AUTH_TOKEN".into(),
                    "ANTHROPIC_BASE_URL".into(),
                    "GEMINI_API_KEY".into(),
                    "GOOGLE_GENERATIVE_AI_API_KEY".into(),
                    "GROQ_API_KEY".into(),
                    "MISTRAL_API_KEY".into(),
                    "OPENROUTER_API_KEY".into(),
                    "TOGETHER_API_KEY".into(),
                    "DEEPSEEK_API_KEY".into(),
                    "XAI_API_KEY".into(),
                    "AZURE_API_KEY".into(),
                    "OPENCODE_CONFIG_DIR".into(),
                ],
                client_config: Some(ClientConfig {
                    file_name: "opencode.json".into(),
                    contents: format!(
                        r#"{{
  "$schema": "https://opencode.ai/config.json",
  "model": "sarathi/{PLACEHOLDER_MODEL}",
  "provider": {{
    "sarathi": {{
      "npm": "@ai-sdk/openai-compatible",
      "name": "Sarathi",
      "options": {{
        "baseURL": "{PLACEHOLDER_BASE_V1}",
        "apiKey": "sarathi-local"
      }},
      "models": {{
        "{PLACEHOLDER_MODEL}": {{
          "name": "{PLACEHOLDER_MODEL_NAME}"
        }}
      }}
    }}
  }}
}}
"#
                    ),
                }),
            },
            user_defined: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_on(port: u16) -> LaunchContext {
        LaunchContext {
            port,
            model_id: "Qwen/Qwen2.5-3B".into(),
            model_name: "Qwen2.5-3B".into(),
            client_dir: r"C:\data\clients\opencode".into(),
        }
    }

    #[test]
    fn placeholders_are_filled_with_the_live_port() {
        let env = HashMap::from([
            ("ANTHROPIC_BASE_URL".to_string(), PLACEHOLDER_BASE.to_string()),
            ("OPENAI_BASE_URL".to_string(), PLACEHOLDER_BASE_V1.to_string()),
            ("OPENAI_API_KEY".to_string(), "sarathi-local".to_string()),
        ]);

        let resolved = resolve_env(&env, &ctx_on(11435));

        assert_eq!(resolved["ANTHROPIC_BASE_URL"], "http://127.0.0.1:11435");
        assert_eq!(resolved["OPENAI_BASE_URL"], "http://127.0.0.1:11435/v1");
        assert_eq!(resolved["OPENAI_API_KEY"], "sarathi-local", "plain values pass through");
    }

    #[test]
    fn a_changed_port_changes_the_address() {
        // The address must never be cached: the port is user-configurable.
        let env = HashMap::from([("BASE".to_string(), PLACEHOLDER_BASE.to_string())]);

        assert_eq!(resolve_env(&env, &ctx_on(11435))["BASE"], "http://127.0.0.1:11435");
        assert_eq!(resolve_env(&env, &ctx_on(9999))["BASE"], "http://127.0.0.1:9999");
    }

    #[test]
    fn the_v1_placeholder_is_not_mangled_by_the_shorter_one() {
        // `{gatewayUrl}` is a prefix of `{gatewayUrlV1}`; replacing in the wrong
        // order would leave a stray "V1" on the end of the address.
        let env = HashMap::from([("B".to_string(), PLACEHOLDER_BASE_V1.to_string())]);

        assert_eq!(resolve_env(&env, &ctx_on(11435))["B"], "http://127.0.0.1:11435/v1");
    }

    #[test]
    fn a_model_id_with_a_slash_is_made_safe_for_clients() {
        // opencode addresses models as `provider/model`, so an id carrying its
        // own slash would be split in the wrong place.
        assert_eq!(ctx_on(1).client_model_id(), "Qwen-Qwen2.5-3B");
    }

    #[test]
    fn the_model_name_placeholder_is_not_mangled_by_the_shorter_one() {
        // `{model}` is a prefix of `{modelName}`, the same trap as the two URLs.
        let env = HashMap::from([("M".to_string(), PLACEHOLDER_MODEL_NAME.to_string())]);

        assert_eq!(resolve_env(&env, &ctx_on(11435))["M"], "Qwen2.5-3B");
    }

    #[test]
    fn generated_config_escapes_windows_paths() {
        // Client configs are JSON and client directories are full of
        // backslashes; substituting one raw would produce an unparseable file.
        let filled = fill_placeholders(r#"{"dir":"{clientDir}"}"#, &ctx_on(11435), true);

        assert_eq!(filled, r#"{"dir":"C:\\data\\clients\\opencode"}"#);
        serde_json::from_str::<serde_json::Value>(&filled).expect("must stay valid JSON");
    }

    #[test]
    fn the_opencode_config_names_sarathi_as_the_provider() {
        let opencode = builtin_tools().into_iter().find(|t| t.id == "opencode").unwrap();
        let config = opencode.launch.client_config.expect("opencode needs a generated config");
        let body = fill_placeholders(&config.contents, &ctx_on(11435), true);

        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(parsed["provider"]["sarathi"]["name"], "Sarathi");
        assert_eq!(parsed["provider"]["sarathi"]["options"]["baseURL"], "http://127.0.0.1:11435/v1");
        assert_eq!(parsed["model"], "sarathi/Qwen-Qwen2.5-3B");
        assert_eq!(parsed["provider"]["sarathi"]["models"]["Qwen-Qwen2.5-3B"]["name"], "Qwen2.5-3B");
    }

    #[test]
    fn every_shipped_tool_strips_the_credentials_that_would_override_sarathi() {
        for tool in builtin_tools() {
            let removed = &tool.launch.env_remove;
            let set: Vec<&String> = tool.launch.env.keys().collect();

            assert!(!removed.is_empty(), "{} strips nothing", tool.id);

            // Setting a variable is not enough on its own: whatever Sarathi does
            // not set, the tool still reads from the user's environment. Each
            // credential its protocol understands must be pinned or removed.
            let credentials: &[&str] = match tool.protocol {
                Protocol::Anthropic => &["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
                Protocol::OpenAi => &["OPENAI_API_KEY", "OPENAI_BASE_URL"],
            };

            for key in credentials {
                assert!(
                    removed.iter().any(|r| r.as_str() == *key)
                        || set.iter().any(|s| s.as_str() == *key),
                    "{} neither sets nor strips {key}, so the user's own would win",
                    tool.id
                );
            }
        }
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
    fn a_tool_that_prints_only_its_version_is_still_identified() {
        // Regression: opencode answers `--version` with "1.18.13" and nothing
        // else, so requiring the name back reported it as never installed.
        assert!(output_identifies_tool("1.18.13", "opencode"));
        assert!(output_identifies_tool("v2.3", "opencode"));
        assert!(output_identifies_tool("1.18.13\n", "opencode"));

        // Still not enough on its own to call an error message a tool.
        assert!(!output_identifies_tool("command not found", "opencode"));
        assert!(!output_identifies_tool("Microsoft Windows [Version 10.0.26200]", "opencode"));
        assert!(!output_identifies_tool("1", "opencode"), "a lone integer is not a version");
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

    /// Everything Sarathi hands a tool: its environment plus any config it
    /// generates. Which of the two carries the address is an implementation
    /// detail — opencode is told through its config precisely because a bare
    /// `OPENAI_BASE_URL` is what made it treat OpenAI as a second provider.
    fn wiring_of(t: &ToolSpec) -> String {
        let env = t.launch.env.values().cloned().collect::<Vec<_>>().join(" ");
        let config = t.launch.client_config.as_ref().map(|c| c.contents.as_str()).unwrap_or("");
        format!("{env} {config}")
    }

    #[test]
    fn each_builtin_points_at_the_endpoint_it_speaks() {
        for t in builtin_tools() {
            let wiring = wiring_of(&t);
            match t.protocol {
                Protocol::Anthropic => assert!(
                    wiring.contains(PLACEHOLDER_BASE),
                    "{} speaks Anthropic but never receives that address",
                    t.id
                ),
                Protocol::OpenAi => assert!(
                    wiring.contains(PLACEHOLDER_BASE_V1),
                    "{} speaks OpenAI but never receives that address",
                    t.id
                ),
            }
        }
    }

    #[test]
    fn anthropic_tools_get_the_base_url_and_openai_tools_get_v1() {
        // The OpenAI endpoint lives under /v1; the Anthropic one does not.
        // Checked on the filled result, so a mis-ordered substitution shows up.
        let ctx = ctx_on(11435);

        for t in builtin_tools() {
            let wiring = fill_placeholders(&wiring_of(&t), &ctx, false);
            match t.protocol {
                Protocol::Anthropic => {
                    assert!(wiring.contains("http://127.0.0.1:11435"), "{}", t.id);
                    assert!(!wiring.contains("/v1"), "{} must not be sent to /v1", t.id);
                }
                Protocol::OpenAi => {
                    assert!(wiring.contains("http://127.0.0.1:11435/v1"), "{}", t.id)
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
