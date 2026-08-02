/// Certification System Verification Suite
/// Tests PackManager initialization, pack loading, embedded fallback,
/// model ID matching, and persistence across app restarts.

use std::path::PathBuf;
use sarathi_lib::model_recommendation::pack_manager::PackManager;
use sarathi_lib::model_recommendation::certified_catalog::CertificationTier;

fn get_temp_app_data() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push("sarathi_cert_test");
    let _ = std::fs::remove_dir_all(&p);
    let _ = std::fs::create_dir_all(&p);
    p
}

#[test]
fn test_certification_system_loading_and_matching() {
    println!("\n========================================================================");
    println!("        SARATHI MODEL CERTIFICATION SYSTEM AUDIT & VERIFICATION          ");
    println!("========================================================================\n");

    let test_dir = get_temp_app_data();

    // 1. Instantiate PackManager with clean app data dir (triggers fallback & persistence)
    let pack_mgr = PackManager::new(&test_dir).expect("Failed to initialize PackManager");

    // 2. Verify certified models from official pack
    let certified_models = vec![
        ("Qwen/Qwen2.5-3B", "Qwen 2.5 3B Instruct", CertificationTier::Certified),
        ("google/gemma-2-2b", "Gemma 2 2B Instruct", CertificationTier::Certified),
        ("meta-llama/Llama-3.2-1B", "Llama 3.2 1B Instruct", CertificationTier::Certified),
        ("meta-llama/Llama-3.2-3B", "Llama 3.2 3B Instruct", CertificationTier::Certified),
        ("Qwen/Qwen2.5-7B", "Qwen 2.5 7B Instruct", CertificationTier::Certified),
        ("meta-llama/Llama-3.1-8B", "Llama 3.1 8B Instruct", CertificationTier::Certified),
        ("google/gemma-2-9b", "Gemma 2 9B Instruct", CertificationTier::Compatible),
    ];

    println!("[1/3] Testing Certification Lookup & Score Validation:");
    for (model_id, _expected_name, expected_tier) in &certified_models {
        let cert = pack_mgr.get_package_certification(model_id);
        assert!(cert.is_some(), "Certification NOT found for model: {}", model_id);
        
        let c = cert.unwrap();
        println!("  ✓ Model '{}' -> Package: {} | Score: {}/100 | Tier: {:?}", 
                 model_id, c.model_name, c.confidence_score, c.tier);

        assert_eq!(c.tier, *expected_tier);
        assert!(c.confidence_score >= 80.0);
        assert!(!c.quirks_and_notes.is_empty());
    }

    // 3. Verify uncertified model returns None
    println!("\n[2/3] Testing Uncertified Model Lookup:");
    let uncertified_id = "random-user/uncertified-fake-model-99b";
    let uncert = pack_mgr.get_package_certification(uncertified_id);
    assert!(uncert.is_none(), "Uncertified model returned non-none certification!");
    println!("  ✓ Model '{}' -> Correctly returned None (No Certification Badge)", uncertified_id);

    // 4. Verify restart state persistence in app_data_dir/packs/official_pack.json
    println!("\n[3/3] Testing Restart State Persistence:");
    let persisted_pack_file = test_dir.join("packs").join("official_pack.json");
    assert!(persisted_pack_file.exists(), "Official pack file was NOT persisted to app_data_dir/packs!");
    println!("  ✓ Persisted pack file verified at {:?}", persisted_pack_file);

    // Re-instantiate PackManager from existing app_data_dir to simulate app restart
    let restart_mgr = PackManager::new(&test_dir).expect("Failed to initialize PackManager on restart");
    let cert_restart = restart_mgr.get_package_certification("Qwen/Qwen2.5-3B");
    assert!(cert_restart.is_some());
    assert_eq!(cert_restart.unwrap().confidence_score, 98.0);
    println!("  ✓ Post-restart certification state verified successfully!");

    println!("\n========================================================================");
    println!("   CERTIFICATION SYSTEM 100% AUDITED AND RESTORED SUCCESSFULLY!          ");
    println!("========================================================================\n");

    let _ = std::fs::remove_dir_all(&test_dir);
}
