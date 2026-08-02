/// Real-World Production Model Switching & Memory Engine Validation Test
/// Performs actual LLM inference loading real 4.2GB GGUF models across model switches & app restarts.

use std::path::PathBuf;
use std::sync::Arc;
use sarathi_lib::ai_engine::manager::InferenceManager;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams};
use sarathi_lib::memory_engine::MemoryManager;

fn get_app_data_dir() -> PathBuf {
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| r"C:\Users\lenovo\AppData\Roaming".to_string());
    PathBuf::from(appdata).join("com.sarathi.app")
}

#[tokio::test]
async fn test_real_world_model_switching_memory_persistence() {
    println!("\n========================================================================");
    println!("   SARATHI REAL-WORLD PRODUCTION MODEL SWITCHING MEMORY VALIDATION       ");
    println!("========================================================================\n");

    let app_data = get_app_data_dir();
    println!("[TEST SETUP] App Data Dir: {:?}", app_data);

    let inference_mgr = Arc::new(InferenceManager::new());
    let memory_mgr = Arc::new(MemoryManager::new(&app_data));

    // Ensure test profile fact exists
    let _ = memory_mgr.set_user_profile_fact("name", "Shreyash Patil", "user_fact");
    println!("[TEST SETUP] Initialized user_profile with key 'name' = 'Shreyash Patil'");

    // ========================================================================
    // TEST A: Load Model A (Qwen 2.5 7B) & Perform Real Inference
    // ========================================================================
    println!("\n------------------------------------------------------------------------");
    println!(" [TEST A] Loading Model A: Qwen/Qwen2.5-7B...");
    println!("------------------------------------------------------------------------");

    let model_a_info = inference_mgr
        .load_installed_model_direct(&app_data, "huggingface", "Qwen/Qwen2.5-7B", "Q4_K_M")
        .expect("Failed to load Model A (Qwen/Qwen2.5-7B)");

    println!("[TEST A SUCCESS] Model A loaded: {} (template='{}')", model_a_info.model_name, model_a_info.chat_template);

    let user_msg = ChatMessage {
        role: "user".to_string(),
        content: "What is my name?".to_string(),
        timestamp: None,
    };
    let messages = vec![user_msg.clone()];

    // Process turn through Memory Engine (extract + inject)
    let _ = memory_mgr.process_user_turn(&user_msg.content, None).await;
    let injected_messages = memory_mgr
        .prepare_injected_messages(&messages, &user_msg.content)
        .await
        .expect("Failed to inject memory for Model A");

    println!("[TEST A INJECTION] Injected {} message(s). System Prompt:\n{}", injected_messages.len(), injected_messages[0].content);

    let mut response_text = String::new();
    let params = GenerationParams {
        temperature: 0.2, // Low temp for factual memory recall
        max_tokens: 100,
        ..Default::default()
    };

    let gen_res = inference_mgr.generate_direct(&injected_messages, &params, |chunk| {
        response_text.push_str(&chunk.text);
    });

    println!("[TEST A RESPONSE] Model A Output:\n\"{}\"", response_text.trim());
    assert!(gen_res.is_ok(), "Model A generation failed: {:?}", gen_res);

    let lower_resp_a = response_text.to_lowercase();
    let has_name_a = lower_resp_a.contains("shreyash") || lower_resp_a.contains("patil");
    println!("[TEST A VERIFICATION] Model A remembered name: {}", has_name_a);
    assert!(has_name_a, "Model A failed to answer user's name! Output: '{}'", response_text);

    // ========================================================================
    // TEST B: Unload Model A -> Load Model B (Qwen 2.5 Coder 7B) & Perform Real Inference
    // ========================================================================
    println!("\n------------------------------------------------------------------------");
    println!(" [TEST B] Unloading Model A & Loading Model B: Qwen/Qwen2.5-Coder-7B...");
    println!("------------------------------------------------------------------------");

    let _ = inference_mgr.unload_active_model_direct();

    let model_b_info = inference_mgr
        .load_installed_model_direct(&app_data, "huggingface", "Qwen/Qwen2.5-Coder-7B", "Q4_0")
        .expect("Failed to load Model B (Qwen/Qwen2.5-Coder-7B)");

    println!("[TEST B SUCCESS] Model B loaded: {} (template='{}')", model_b_info.model_name, model_b_info.chat_template);

    let injected_messages_b = memory_mgr
        .prepare_injected_messages(&messages, &user_msg.content)
        .await
        .expect("Failed to inject memory for Model B");

    let mut response_text_b = String::new();
    let gen_res_b = inference_mgr.generate_direct(&injected_messages_b, &params, |chunk| {
        response_text_b.push_str(&chunk.text);
    });

    println!("[TEST B RESPONSE] Model B Output:\n\"{}\"", response_text_b.trim());
    assert!(gen_res_b.is_ok(), "Model B generation failed: {:?}", gen_res_b);

    let lower_resp_b = response_text_b.to_lowercase();
    let has_name_b = lower_resp_b.contains("shreyash") || lower_resp_b.contains("patil");
    println!("[TEST B VERIFICATION] Model B remembered name: {}", has_name_b);
    assert!(has_name_b, "Model B failed to recall user's name after model switch! Output: '{}'", response_text_b);

    // ========================================================================
    // TEST C: Unload Model B -> Load Model C (meta-llama/Llama-3.2-1B)
    // ========================================================================
    println!("\n------------------------------------------------------------------------");
    println!(" [TEST C] Unloading Model B & Loading Model C: meta-llama/Llama-3.2-1B...");
    println!("------------------------------------------------------------------------");

    let _ = inference_mgr.unload_active_model_direct();

    let model_c_info = inference_mgr
        .load_installed_model_direct(&app_data, "huggingface", "meta-llama/Llama-3.2-1B", "Q4_K_M")
        .expect("Failed to load Model C (meta-llama/Llama-3.2-1B)");

    println!("[TEST C SUCCESS] Model C loaded: {} (template='{}')", model_c_info.model_name, model_c_info.chat_template);

    let injected_messages_c = memory_mgr
        .prepare_injected_messages(&messages, &user_msg.content)
        .await
        .expect("Failed to inject memory for Model C");

    let mut response_text_c = String::new();
    let gen_res_c = inference_mgr.generate_direct(&injected_messages_c, &params, |chunk| {
        response_text_c.push_str(&chunk.text);
    });

    println!("[TEST C RESPONSE] Model C Output:\n\"{}\"", response_text_c.trim());
    assert!(gen_res_c.is_ok(), "Model C generation failed: {:?}", gen_res_c);

    let lower_resp_c = response_text_c.to_lowercase();
    let has_name_c = lower_resp_c.contains("shreyash") || lower_resp_c.contains("patil");
    println!("[TEST C VERIFICATION] Model C remembered name: {}", has_name_c);
    assert!(has_name_c, "Model C failed to recall user's name after 2nd model switch! Output: '{}'", response_text_c);

    // ========================================================================
    // TEST D: Re-initialize Memory & Inference Managers (App Restart Simulation)
    // ========================================================================
    println!("\n------------------------------------------------------------------------");
    println!(" [TEST D] Simulating Complete App Restart (New Manager Instances)...");
    println!("------------------------------------------------------------------------");

    drop(inference_mgr);
    drop(memory_mgr);

    let fresh_inference_mgr = Arc::new(InferenceManager::new());
    let fresh_memory_mgr = Arc::new(MemoryManager::new(&app_data));

    let model_restart_info = fresh_inference_mgr
        .load_installed_model_direct(&app_data, "huggingface", "Qwen/Qwen2.5-Coder-7B", "Q4_0")
        .expect("Failed to load model after app restart");

    println!("[TEST D SUCCESS] Loaded model after restart: {}", model_restart_info.model_name);

    let injected_messages_d = fresh_memory_mgr
        .prepare_injected_messages(&messages, &user_msg.content)
        .await
        .expect("Failed to inject memory after app restart");

    let mut response_text_d = String::new();
    let gen_res_d = fresh_inference_mgr.generate_direct(&injected_messages_d, &params, |chunk| {
        response_text_d.push_str(&chunk.text);
    });

    println!("[TEST D RESPONSE] Model Output After Restart:\n\"{}\"", response_text_d.trim());
    assert!(gen_res_d.is_ok(), "Generation failed after restart: {:?}", gen_res_d);

    let lower_resp_d = response_text_d.to_lowercase();
    let has_name_d = lower_resp_d.contains("shreyash") || lower_resp_d.contains("patil");
    println!("[TEST D VERIFICATION] Memory survived application restart: {}", has_name_d);
    assert!(has_name_d, "Memory failed to survive app restart! Output: '{}'", response_text_d);

    println!("\n========================================================================");
    println!("   ALL 4 REAL-WORLD MODEL SWITCHING TESTS PASSED 100%!                   ");
    println!("========================================================================\n");
}
