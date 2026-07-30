//! OS details collector using sysinfo and platform utilities

use crate::system_analyzer::process_utils::create_hidden_command;
use crate::system_analyzer::traits::OsInfo;
use sysinfo::System;

/// Detects operating system information
pub fn detect_os() -> OsInfo {
    let name = System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
    let version = System::os_version().unwrap_or_else(|| "Unknown".to_string());
    let build_number = System::kernel_version().unwrap_or_else(|| "Unknown".to_string());
    let architecture = std::env::consts::ARCH.to_string();

    let (edition, locale) = query_os_edition_and_locale();

    OsInfo {
        name,
        edition,
        version,
        build_number,
        architecture,
        locale,
    }
}

fn query_os_edition_and_locale() -> (String, String) {
    let mut edition = "Standard".to_string();
    let mut locale = "en-US".to_string();

    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = create_hidden_command("wmic")
            .args(["os", "get", "Caption,MUILanguages", "/format:list"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Some((k, v)) = line.trim().split_once('=') {
                    match k.trim() {
                        "Caption" => {
                            let cap = v.trim();
                            if !cap.is_empty() {
                                edition = cap.to_string();
                            }
                        }
                        "MUILanguages" => {
                            let loc = v.trim();
                            if !loc.is_empty() {
                                locale = loc.to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    (edition, locale)
}
