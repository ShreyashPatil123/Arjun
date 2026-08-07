//! IPC for the Launch screen.
//!
//! Every command reports *why* something cannot happen rather than returning a
//! bare failure — the screen is aimed at people who will not read a log file,
//! so "npm is needed to install this, but it is not on this machine" has to
//! reach the UI intact.

use std::sync::Arc;

use tauri::{AppHandle, Emitter, Manager, State};

use crate::gateway::state::{GatewayState, GatewayStats};
use crate::launcher::{
    self, registry, spec::ToolSpec, LaunchedProcesses, ToolState, ToolStatus,
};

/// Everything the Launch screen renders in one call.
///
/// Bundled deliberately: the screen shows tools and server status together, and
/// two round trips could disagree — a tool marked Running beside a server shown
/// as stopped.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchOverview {
    pub tools: Vec<ToolStatus>,
    pub gateway: GatewayStats,
    /// Problems loading user tool definitions, surfaced rather than swallowed.
    pub warnings: Vec<String>,
    /// False when no model is loaded. Launching then produces a tool that
    /// connects and fails on its first question, so the UI disables Launch.
    pub can_launch: bool,
    /// Why launching is unavailable, when it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_reason: Option<String>,
}

fn resolve_registry(app: &AppHandle) -> registry::Registry {
    match app.path().app_data_dir() {
        Ok(dir) => registry::load(&dir),
        // Without an app data directory there are no user entries, only the
        // shipped ones — still a usable screen.
        Err(_) => registry::Registry {
            tools: launcher::spec::builtin_tools(),
            warnings: vec!["Could not read the app data folder; custom tools were not loaded.".into()],
        },
    }
}

fn status_for(spec: &ToolSpec, procs: &LaunchedProcesses) -> ToolStatus {
    // A process Sarathi started takes precedence: it is the most specific thing
    // we know, and re-detecting would only report "installed".
    let state = match procs.pid_of(&spec.id) {
        Some(pid) => ToolState::Running { pid },
        None => launcher::detect(spec),
    };

    // Only offer an install command when it could actually run.
    let install_command = spec.install.as_ref().and_then(|i| {
        launcher::manager_available(i.manager).then(|| i.manager.command_line(&i.package))
    });

    ToolStatus {
        id: spec.id.clone(),
        name: spec.name.clone(),
        description: spec.description.clone(),
        protocol: spec.protocol,
        user_defined: spec.user_defined,
        install_command,
        state,
    }
}

#[tauri::command]
pub async fn get_launch_overview(
    app: AppHandle,
    gateway: State<'_, Arc<GatewayState>>,
    procs: State<'_, Arc<LaunchedProcesses>>,
) -> Result<LaunchOverview, String> {
    let reg = resolve_registry(&app);
    let procs = procs.inner().clone();

    // Detection runs external programs, so keep it off the async runtime.
    let specs = reg.tools.clone();
    let tools = tokio::task::spawn_blocking(move || {
        specs.iter().map(|s| status_for(s, &procs)).collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("could not check installed tools: {e}"))?;

    let model = gateway.inference.get_loaded_model_info();
    let can_launch = model.is_some();

    // Reported from whether the server is actually up, not assumed.
    //
    // The handle is only managed once a listener is bound, so its presence is
    // the fact rather than an inference. Passing `true` unconditionally meant
    // the dashboard read "Running" while nothing was listening — which is
    // exactly the state a client sees as ConnectionRefused, with the one screen
    // that could have explained it insisting the server was fine.
    let gateway_running = app.try_state::<crate::gateway::GatewayHandle>().is_some();

    Ok(LaunchOverview {
        tools,
        gateway: gateway.stats(gateway_running),
        warnings: reg.warnings,
        can_launch,
        blocked_reason: (!can_launch)
            .then(|| "Load a model first — tools would connect but get no answer.".to_string()),
    })
}

/// The exact command an install would run, for the confirmation prompt.
///
/// Separate from [`install_tool`] so the UI can show it and wait for consent.
/// Nothing is executed here.
#[tauri::command]
pub async fn preview_tool_install(app: AppHandle, tool_id: String) -> Result<String, String> {
    let reg = resolve_registry(&app);
    let spec = reg
        .tools
        .iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| format!("no tool called '{tool_id}'"))?;

    let install = spec
        .install
        .as_ref()
        .ok_or_else(|| format!("{} cannot be installed by Sarathi", spec.name))?;

    if !launcher::manager_available(install.manager) {
        return Err(format!(
            "{} is needed to install {}, but it is not available on this machine.",
            install.manager.program(),
            spec.name
        ));
    }

    Ok(install.manager.command_line(&install.package))
}

#[tauri::command]
pub async fn install_tool(app: AppHandle, tool_id: String) -> Result<(), String> {
    let reg = resolve_registry(&app);
    let spec = reg
        .tools
        .iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| format!("no tool called '{tool_id}'"))?
        .clone();

    // Installs are slow and run an external program. Each line the package
    // manager prints is forwarded to the UI as it arrives, so the card can show
    // what is actually happening instead of a spinner with no end in sight.
    let emitter = app.clone();
    let progress_id = tool_id.clone();

    tokio::task::spawn_blocking(move || {
        launcher::install(&spec, |line| {
            let _ = emitter.emit(
                "tool:install-progress",
                serde_json::json!({ "toolId": progress_id, "line": line }),
            );
        })
    })
    .await
    .map_err(|e| format!("install task failed: {e}"))?
}

#[tauri::command]
pub async fn launch_tool(
    app: AppHandle,
    gateway: State<'_, Arc<GatewayState>>,
    procs: State<'_, Arc<LaunchedProcesses>>,
    tool_id: String,
) -> Result<u32, String> {
    let Some(model) = gateway.inference.get_loaded_model_info() else {
        return Err("Load a model first — the tool would connect but get no answer.".into());
    };

    let reg = resolve_registry(&app);
    let spec = reg
        .tools
        .iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| format!("no tool called '{tool_id}'"))?
        .clone();

    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not locate the Sarathi data directory: {e}"))?;

    // One workspace shared by every tool, and a private config directory per
    // tool. Two tools launched together therefore open on the same files while
    // keeping their own provider state isolated from each other and from the
    // user's own installs.
    let workspace = data_dir.join("workspace");
    let client_dir = data_dir.join("clients").join(&tool_id);

    // The live port, not the configured one: they differ if the configured
    // port was taken and the gateway bound elsewhere. Model likewise comes from
    // what is loaded right now, so the tool cannot open on a stale one.
    let port = gateway.port();
    let ctx = launcher::spec::LaunchContext {
        port,
        model_id: model.model_id.clone(),
        model_name: model.model_name.clone(),
        client_dir: client_dir.to_string_lossy().to_string(),
        context_length: model.context_length,
    };
    let model_label = model.model_name.clone();

    let pid = tokio::task::spawn_blocking(move || launcher::launch(&spec, &ctx, &workspace))
        .await
        .map_err(|e| format!("launch task failed: {e}"))??;

    procs.record(&tool_id, pid);
    log::info!(
        "[LAUNCH] Started '{tool_id}' (pid {pid}) as a Sarathi client on port {port}, serving '{model_label}'"
    );
    Ok(pid)
}

/// Stops tracking a tool.
///
/// Sarathi does not kill the process: these are terminals and editors the user
/// may be working in, and closing one from another window would lose their work.
/// The card returns to Ready.
#[tauri::command]
pub async fn forget_tool_process(
    procs: State<'_, Arc<LaunchedProcesses>>,
    tool_id: String,
) -> Result<(), String> {
    procs.forget(&tool_id);
    Ok(())
}

/// Path of the file where custom tools are defined, so the UI can point at it.
#[tauri::command]
pub async fn user_tools_file(app: AppHandle) -> Result<String, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("could not resolve the app data folder: {e}"))?;
    Ok(registry::user_tools_path(&dir).to_string_lossy().to_string())
}
