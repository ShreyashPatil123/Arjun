//! The embedded browser's own network traffic.
//!
//! ARJUN's window is Microsoft Edge WebView2. The runtime is not ARJUN's code,
//! and with its defaults it contacts Microsoft on its own schedule, whatever
//! the page does. Measured on runtime 153.0.4234.48, online, from a single
//! launch of ARJUN:
//!
//! - `edge.microsoft.com/componentupdater/…`, from the network service about a
//!   minute after launch (component updates);
//! - a `wpad` lookup (proxy auto-discovery);
//! - `substrate.office.com` (Microsoft 365's profile service), from the browser
//!   process itself: `oneauth.dll` fetches the Windows account's profile and
//!   picture. It uses its own HTTP client, not Chromium's network stack, so no
//!   network switch reaches it. Only turning off the feature that loads OneAuth
//!   stops it.
//!
//! The page cannot cause any of this, because its CSP allows only `self` and
//! ARJUN's IPC. The fix therefore goes into how the window's runtime is launched:
//! `additionalBrowserArgs` in `tauri.conf.json`. Setting that field replaces
//! wry's defaults, which is why they are repeated there.
//!
//! The tests below pin each switch with the reason it is there, so removing one
//! is a deliberate change rather than an accident.
//!
//! Limits: feature names such as `msOneAuthWAM` belong to the runtime and can
//! change when it updates. The resolver rule fails closed for anything sent
//! through Chromium's network stack, whatever it is called. The browser
//! process's own HTTP clients are covered only by the feature switch, so each
//! new runtime version has to be measured again (docs/sih/webview2-egress.md).

/// Runtime features the main window must be launched with disabled, and why.
const DISABLED_FEATURES: &[(&str, &str)] = &[
    ("msWebOOUI", "wry default: the text-selection mini menu"),
    ("msPdfOOUI", "wry default: the PDF mini menu"),
    ("msSmartScreenProtection", "wry default: SmartScreen sends visited URLs to Microsoft"),
    (
        "msOneAuthWAM",
        "loads oneauth.dll, which fetches the Windows account's profile from substrate.office.com at launch",
    ),
];

/// Switches the main window must be launched with, and why.
const REQUIRED_SWITCHES: &[(&str, &str)] = &[
    ("--disable-component-update", "polls edge.microsoft.com/componentupdater"),
    ("--disable-background-networking", "variations seed and other background fetches"),
    ("--no-proxy-server", "stops the wpad proxy auto-discovery lookup"),
    (
        "--host-resolver-rules",
        "fails every non-loopback hostname inside the WebView before any DNS query",
    ),
];

/// Hostnames the resolver rule may leave resolvable. Every one of them resolves
/// to this machine: the page itself is served from `tauri.localhost` and
/// `ipc.localhost`.
const LOOPBACK_EXCLUSIONS: &[&str] = &["localhost", "*.localhost", "127.0.0.1"];

fn configured_args() -> String {
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../../tauri.conf.json")).expect("tauri.conf.json parses");
    let windows = config["app"]["windows"].as_array().expect("app.windows is a list");
    let main = windows
        .iter()
        .find(|w| w["label"] == "main")
        .expect("a window labelled main");
    main["additionalBrowserArgs"]
        .as_str()
        .expect("the main window sets additionalBrowserArgs")
        .to_string()
}

/// Splits a command line on spaces outside double quotes, dropping the quotes.
fn split_args(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            ' ' if !quoted => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn value_of<'a>(args: &'a [String], switch: &str) -> Option<&'a str> {
    let prefix = format!("{switch}=");
    args.iter().find_map(|a| a.strip_prefix(prefix.as_str()))
}

#[test]
fn the_quoted_resolver_rule_stays_one_argument() {
    let args = split_args(r#"--a --rules="MAP * ~NOTFOUND, EXCLUDE localhost" --b"#);
    assert_eq!(args, vec!["--a", "--rules=MAP * ~NOTFOUND, EXCLUDE localhost", "--b"]);
}

#[test]
fn every_runtime_feature_that_calls_microsoft_is_disabled() {
    let args = split_args(&configured_args());
    let disabled: Vec<&str> = value_of(&args, "--disable-features")
        .expect("--disable-features is set")
        .split(',')
        .collect();
    for (feature, why) in DISABLED_FEATURES {
        assert!(disabled.contains(feature), "{feature} must stay disabled: {why}");
    }
}

#[test]
fn every_switch_that_stops_background_traffic_is_present() {
    let args = split_args(&configured_args());
    for (switch, why) in REQUIRED_SWITCHES {
        let present = args.iter().any(|a| a == switch || a.starts_with(&format!("{switch}=")));
        assert!(present, "{switch} must stay set: {why}");
    }
}

/// The resolver rule is only worth having if it fails everything by default
/// and lets through nothing but this machine.
#[test]
fn the_resolver_rule_fails_closed_and_exempts_only_loopback() {
    let args = split_args(&configured_args());
    let rule = value_of(&args, "--host-resolver-rules").expect("--host-resolver-rules is set");
    let parts: Vec<&str> = rule.split(',').map(str::trim).collect();
    assert_eq!(parts.first(), Some(&"MAP * ~NOTFOUND"), "the first rule maps every host to not-found");
    for part in &parts[1..] {
        let host = part
            .strip_prefix("EXCLUDE ")
            .unwrap_or_else(|| panic!("only EXCLUDE rules may follow the catch-all, found {part:?}"));
        assert!(
            LOOPBACK_EXCLUSIONS.contains(&host),
            "{host} is not this machine, so it must not be exempt from the catch-all"
        );
    }
}
