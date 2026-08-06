//! Central PackManager Abstraction
//! Manages pluggable certification packs, runtime profiles, and security validation.

use crate::model_recommendation::certified_catalog::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub struct PackManager {
    app_data_dir: PathBuf,
    packs: Arc<Mutex<Vec<CertificationPack>>>,
    runtime_profiles: Arc<Mutex<HashMap<String, RuntimeProfile>>>,
}

impl PackManager {
    pub fn new(app_data_dir: &Path) -> Result<Self> {
        let manager = Self {
            app_data_dir: app_data_dir.to_path_buf(),
            packs: Arc::new(Mutex::new(Vec::new())),
            runtime_profiles: Arc::new(Mutex::new(HashMap::new())),
        };

        manager.reload_all_packs()?;
        Ok(manager)
    }

    /// Loads all certification packs from sidecars/packs and app_data_dir/packs
    pub fn reload_all_packs(&self) -> Result<()> {
        let cwd = std::env::current_dir().unwrap_or_default();
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();

        let candidate_dirs = vec![
            self.app_data_dir.join("packs"),
            cwd.join("sidecars").join("packs"),
            cwd.join("src-tauri").join("sidecars").join("packs"),
            cwd.parent().map(|p| p.join("sidecars").join("packs")).unwrap_or_default(),
            exe_dir.join("sidecars").join("packs"),
            exe_dir.join("packs"),
        ];

        let mut loaded_packs = Vec::new();
        for dir in candidate_dirs {
            if dir.exists() && dir.is_dir() {
                if let Ok(entries) = fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("json") {
                            if let Ok(content) = fs::read_to_string(&path) {
                                if let Ok(pack) = serde_json::from_str::<CertificationPack>(&content) {
                                    log::info!("[PACK_MANAGER] Loaded certification pack '{}' from {:?}", pack.name, path);
                                    if !loaded_packs.iter().any(|p: &CertificationPack| p.pack_id == pack.pack_id) {
                                        loaded_packs.push(pack);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Always ensure app_data_dir/packs contains official_pack.json for cross-restart state persistence
        let target_dir = self.app_data_dir.join("packs");
        let target_file = target_dir.join("official_pack.json");
        if !target_file.exists() {
            let embedded = include_str!("../../../sidecars/packs/official_pack.json");
            let _ = fs::create_dir_all(&target_dir);
            let _ = fs::write(&target_file, embedded);
            log::info!("[PACK_MANAGER] Seeded official_pack.json into app_data_dir/packs");
        }

        // Fallback: If no packs found on disk, parse embedded official_pack.json
        if loaded_packs.is_empty() {
            let embedded = include_str!("../../../sidecars/packs/official_pack.json");
            if let Ok(pack) = serde_json::from_str::<CertificationPack>(embedded) {
                log::info!("[PACK_MANAGER] Loaded fallback embedded certification pack '{}'", pack.name);
                loaded_packs.push(pack);
            }
        }

        let mut lock = self.packs.lock().unwrap();
        *lock = loaded_packs;

        // Reload decoupled runtime profiles
        self.reload_runtime_profiles()?;

        Ok(())
    }

    fn reload_runtime_profiles(&self) -> Result<()> {
        let cwd = std::env::current_dir().unwrap_or_default();
        let candidate_dirs = vec![
            self.app_data_dir.join("runtime_profiles"),
            cwd.join("sidecars").join("runtime_profiles"),
            cwd.join("src-tauri").join("sidecars").join("runtime_profiles"),
            cwd.parent().map(|p| p.join("sidecars").join("runtime_profiles")).unwrap_or_default(),
            // Next to the executable, for an installed app: its working
            // directory is wherever the shortcut points, never the repo.
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|dir| dir.join("sidecars").join("runtime_profiles")))
                .unwrap_or_default(),
        ];

        let mut profiles = HashMap::new();
        for dir in candidate_dirs {
            if dir.exists() && dir.is_dir() {
                if let Ok(entries) = fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("json") {
                            if let Ok(content) = fs::read_to_string(&path) {
                                if let Ok(profile) = serde_json::from_str::<RuntimeProfile>(&content) {
                                    log::info!("[PACK_MANAGER] Loaded runtime profile '{}' from {:?}", profile.profile_id, path);
                                    profiles.insert(profile.profile_id.clone(), profile);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut lock = self.runtime_profiles.lock().unwrap();
        *lock = profiles;

        Ok(())
    }

    pub fn get_package_certification(&self, model_id: &str) -> Option<PackageCertification> {
        let packs = self.packs.lock().unwrap();
        let target_norm = model_id.to_lowercase().replace(['/', '-', '_', ' '], "");
        
        for pack in packs.iter() {
            for pkg in &pack.certified_packages {
                let pkg_model_norm = pkg.model_id.to_lowercase().replace(['/', '-', '_', ' '], "");
                let pkg_id_norm = pkg.package_id.to_lowercase().replace(['/', '-', '_', ' '], "");
                if pkg_model_norm == target_norm || target_norm.contains(&pkg_model_norm) || pkg_model_norm.contains(&target_norm) || pkg_id_norm.contains(&target_norm) {
                    return Some(pkg.clone());
                }
            }
        }
        None
    }

    pub fn get_runtime_profile(&self, profile_id: &str) -> Option<RuntimeProfile> {
        let profiles = self.runtime_profiles.lock().unwrap();
        profiles.get(profile_id).cloned()
    }

    pub fn get_all_certifications(&self) -> Vec<PackageCertification> {
        let packs = self.packs.lock().unwrap();
        let mut list = Vec::new();
        for pack in packs.iter() {
            list.extend(pack.certified_packages.clone());
        }
        list
    }

    pub fn get_recommended_models(&self) -> Vec<PackageCertification> {
        self.get_all_certifications()
            .into_iter()
            .filter(|c| c.tier == CertificationTier::Certified)
            .collect()
    }

    pub fn get_compatible_models(&self) -> Vec<PackageCertification> {
        self.get_all_certifications()
            .into_iter()
            .filter(|c| c.tier == CertificationTier::Compatible)
            .collect()
    }

    pub fn get_experimental_models(&self) -> Vec<PackageCertification> {
        self.get_all_certifications()
            .into_iter()
            .filter(|c| c.tier == CertificationTier::Experimental)
            .collect()
    }
}
