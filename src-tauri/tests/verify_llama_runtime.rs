use std::path::PathBuf;
use sarathi_lib::ai_engine::manager::InferenceManager;

#[test]
fn test_llama_runtime_all_certified_models() {
    println!("========================================================================");
    println!("  SARATHI PRODUCTION STAGE 4 RUNTIME AUDIT (ALL 4 CERTIFIED MODELS)    ");
    println!("========================================================================");

    let appdata_str = std::env::var("APPDATA").unwrap_or_else(|_| r"C:\Users\lenovo\AppData\Roaming".to_string());
    let app_data = PathBuf::from(appdata_str).join("com.sarathi.app");

    println!("App Data Directory: {:?}", app_data);

    let mgr = InferenceManager::new();

    // Testing ALL 4 certified base models
    let models_to_test = vec![
        ("huggingface", "Qwen/Qwen2.5-7B", "Q4_0"),
        ("huggingface", "Qwen/Qwen2.5-Coder-7B", "Q4_0"),
        ("huggingface", "Qwen/Qwen2.5-3B", "Q4_0"),
        ("huggingface", "meta-llama/Llama-3.2-1B", "Q4_0"),
    ];

    for (provider_id, model_id, quantization) in models_to_test {
        println!("\n------------------------------------------------------------------------");
        println!("Testing Production Load & Runtime Execution: '{}' ({})", model_id, quantization);
        println!("------------------------------------------------------------------------");

        // 1. Initial Load
        let load_res = mgr.load_installed_model_direct(
            &app_data,
            provider_id,
            model_id,
            quantization,
        );

        match load_res {
            Ok(info) => {
                println!("✓ [STAGE 4 PASS] Loaded: {} ({}) via {}", info.model_name, info.quantization, info.backend_used);
                println!("   File Path: {}", info.file_path);
                println!("   Context Length: {}", info.context_length);
                println!("   GPU Layers: {}", info.gpu_layers);
                println!("   Threads: {}", info.threads);

                // 2. Unload
                println!("\nTesting Model Unload...");
                assert!(mgr.unload_active_model_direct().is_ok(), "Unload failed!");
                println!("✓ [UNLOAD PASS] Unloaded cleanly.");

                // 3. Reload
                println!("\nTesting Model Reload...");
                let reload_res = mgr.load_installed_model_direct(
                    &app_data,
                    provider_id,
                    model_id,
                    quantization,
                );
                assert!(reload_res.is_ok(), "Reload failed!");
                println!("✓ [RELOAD PASS] Reloaded cleanly.");

                // 4. Final Unload
                assert!(mgr.unload_active_model_direct().is_ok(), "Final unload failed!");
            }
            Err(e) => {
                println!("❌ [LOAD FAILED] Model '{}' failed during Stage 4 Runtime Load: {:?}", model_id, e);
                panic!("Stage 4 Load Failed for model {}: {:?}", model_id, e);
            }
        }
    }

    println!("\n========================================================================");
    println!(" 🎉 ALL 4 CERTIFIED MODELS PASSED STAGE 4 RUNTIME INITIALIZATION & RELOAD ");
    println!("========================================================================");
}
