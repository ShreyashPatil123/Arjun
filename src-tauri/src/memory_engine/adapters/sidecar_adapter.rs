//! Single Sidecar Adapter — Stdio NDJSON-RPC Client
//! Spawns and communicates with the Python sidecar process over stdin/stdout pipes.
//! ZERO firewall prompts, ZERO port conflicts, 100% process lifetime coupling.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};

pub struct SidecarAdapter {
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<std::process::ChildStdin>>,
    reader: Arc<Mutex<BufReader<std::process::ChildStdout>>>,
    request_id: Arc<Mutex<u64>>,
}

impl SidecarAdapter {
    /// Spawns the embedded Python Memory Engine Sidecar
    pub fn spawn(app_data_dir: &std::path::Path) -> Result<Self> {
        let cwd = std::env::current_dir().unwrap_or_default();
        let candidate_paths = vec![
            app_data_dir.join("sidecars").join("memory_engine_sidecar").join("main.py"),
            cwd.join("sidecars").join("memory_engine_sidecar").join("main.py"),
            cwd.join("src-tauri").join("sidecars").join("memory_engine_sidecar").join("main.py"),
            cwd.parent().map(|p| p.join("sidecars").join("memory_engine_sidecar").join("main.py")).unwrap_or_default(),
            std::env::current_exe().ok().and_then(|p| p.parent().map(|dir| dir.join("sidecars").join("memory_engine_sidecar").join("main.py"))).unwrap_or_default(),
        ];

        let target_script = candidate_paths
            .into_iter()
            .find(|p| p.as_os_str().len() > 0 && p.exists())
            .ok_or_else(|| anyhow!("Memory sidecar main.py script not found in any candidate path from CWD '{:?}'", cwd))?;

        log::info!("[SIDECAR] Spawning Python sidecar from {:?}", target_script);

        let mut command = Command::new("python");
        command.arg(&target_script);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::inherit());

        let mut child = command.spawn().map_err(|e| anyhow!("Failed to spawn sidecar python process: {}", e))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("Failed to open sidecar stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to open sidecar stdout"))?;

        let reader = BufReader::new(stdout);

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            reader: Arc::new(Mutex::new(reader)),
            request_id: Arc::new(Mutex::new(1)),
        })
    }

    /// Sends NDJSON-RPC request to Python sidecar over stdin and reads response frame from stdout
    pub fn call_rpc(&self, method: &str, params: Value) -> Result<Value> {
        let req_id = {
            let mut id_lock = self.request_id.lock().unwrap();
            *id_lock += 1;
            *id_lock
        };

        let req_payload = json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        });

        let mut line = serde_json::to_string(&req_payload)?;
        line.push('\n');

        {
            let mut stdin = self.stdin.lock().unwrap();
            stdin.write_all(line.as_bytes())?;
            stdin.flush()?;
        }

        let mut response_line = String::new();
        {
            let mut reader = self.reader.lock().unwrap();
            reader.read_line(&response_line)?;
        }

        if response_line.trim().is_empty() {
            return Err(anyhow!("Received empty EOF response from sidecar stdio stream"));
        }

        let res_val: Value = serde_json::from_str(&response_line)?;
        if let Some(err) = res_val.get("error") {
            return Err(anyhow!("Sidecar RPC error: {}", err["message"].as_str().unwrap_or("Unknown error")));
        }

        Ok(res_val.get("result").cloned().unwrap_or(Value::Null))
    }
}

impl Drop for SidecarAdapter {
    fn drop(&mut self) {
        log::info!("[SIDECAR] Terminating Python sidecar child process");
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
    }
}
