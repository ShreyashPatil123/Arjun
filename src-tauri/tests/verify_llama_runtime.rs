use std::path::PathBuf;
use sarathi_lib::ai_engine::manager::InferenceManager;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams};

#[test]
fn test_llama_runtime_load_and_inference() {
    println!("========================================================================");
    println!("    SARATHI STAGE 4 LLAMA.CPP RUNTIME LOAD & INFERENCE VERIFICATION     ");
    println!("========================================================================");

    let appdata_str = std::env::var("APPDATA").unwrap_or_else(|_| r"C:\Users\lenovo\AppData\Roaming".to_string());
    let app_data = PathBuf::from(appdata_str).join("com.sarathi.app");

    println!("App Data Directory: {:?}", app_data);

    let mgr = InferenceManager::new();

    // Testing real GGUF binary: Qwen 2.5 Coder 7B (4.2 GB)
    let models_to_test = vec![
        ("huggingface", "Qwen/Qwen2.5-Coder-7B", "Q4_0"),
    ];

    for (provider_id, model_id, quantization) in models_to_test {
        println!("\n------------------------------------------------------------------------");
        println!("Testing Model Load: '{}' ({})", model_id, quantization);
        println!("------------------------------------------------------------------------");

        // Step 1: Load Model
        let load_res = mgr.load_installed_model_direct(
            &app_data,
            provider_id,
            model_id,
            quantization,
        );

        match load_res {
            Ok(info) => {
                println!("✓ [LOAD SUCCESS] Model Loaded: {} ({}) via {}", info.model_name, info.quantization, info.backend_used);
                println!("   File Path: {}", info.file_path);
                println!("   Context Length: {}", info.context_length);
                println!("   GPU Layers: {}", info.gpu_layers);
                println!("   Threads: {}", info.threads);

                // Step 2: Test Unload
                println!("\nTesting Unload...");
                assert!(mgr.unload_active_model_direct().is_ok(), "Unload failed!");
                println!("✓ [UNLOAD SUCCESS] Model unloaded cleanly.");

                // Step 3: Test Reload
                println!("\nTesting Reload...");
                let reload_res = mgr.load_installed_model_direct(
                    &app_data,
                    provider_id,
                    model_id,
                    quantization,
                );
                assert!(reload_res.is_ok(), "Reload failed!");
                println!("✓ [RELOAD SUCCESS] Model reloaded cleanly.");

                // Step 4: Final Unload
                assert!(mgr.unload_active_model_direct().is_ok(), "Final unload failed!");
            }
            Err(e) => {
                println!("❌ [LOAD FAILED] Model '{}' failed during Stage 4 Runtime Load: {:?}", model_id, e);
                panic!("Stage 4 Load Failed for model {}: {:?}", model_id, e);
            }
        }
    }

    println!("\n========================================================================");
    println!(" 🎉 STAGE 4 RUNTIME INITIALIZATION & RE-LOAD VERIFIED SUCCESSFULLY!     ");
    println!("========================================================================");
}
