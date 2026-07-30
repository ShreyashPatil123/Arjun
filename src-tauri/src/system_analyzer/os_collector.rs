//! OS details collector using sysinfo and PowerShell CIM queries

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::OsInfo;
use serde::Deserialize;
use sysinfo::System;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct CimOsInfo {
    caption: Option<String>,
    version: Option<String>,
    build_number: Option<String>,
}

/// Detects operating system information
pub fn detect_os() -> OsInfo {
    log::info!("[SYSTEM ANALYZER DEBUG] 🚀 OS Collector Started");

    let mut name = System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
    let mut version = System::os_version().unwrap_or_else(|| "Unknown".to_string());
    let mut build_number = System::kernel_version().unwrap_or_else(|| "Unknown".to_string());
    let mut edition = "Standard".to_string();
    let locale = "en-US".to_string();

    #[cfg(target_os = "windows")]
    {
        if let Ok(cim_os) = query_cim_os() {
            log::info!("[SYSTEM ANALYZER DEBUG] ✓ CIM OS query returned: {:?}", cim_os);
            if let Some(cap) = cim_os.caption {
                edition = cap.trim().to_string();
                name = "Windows".to_string();
            }
            if let Some(v) = cim_os.version {
                version = v.trim().to_string();
            }
            if let Some(b) = cim_os.build_number {
                build_number = b.trim().to_string();
            }
        }
    }

    let architecture = std::env::consts::ARCH.to_string();

    log::info!("[SYSTEM ANALYZER DEBUG] ✓ OS Detection Finished: Name={}, Edition={}, Version={}, Build={}",
        name, edition, version, build_number);

    OsInfo {
        name,
        edition,
        version,
        build_number,
        architecture,
        locale,
    }
}

#[cfg(target_os = "windows")]
fn query_cim_os() -> Result<CimOsInfo, String> {
    let output = create_hidden_command("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber | ConvertTo-Json",
        ])
        .output()
        .map_err(|e| format!("Failed powershell: {}", e))?;

    if !output.status.success() {
        return Err("powershell Get-CimInstance Win32_OperatingSystem failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("empty stdout from powershell".to_string());
    }

    if let Ok(item) = serde_json::from_str::<CimOsInfo>(&stdout) {
        Ok(item)
    } else if let Ok(list) = serde_json::from_str::<Vec<CimOsInfo>>(&stdout) {
        if let Some(first) = list.into_iter().next() {
            Ok(first)
        } else {
            Err("Empty OS list".to_string())
        }
    } else {
        Err(format!("Failed to parse OS CIM JSON: {}", stdout))
    }
}
