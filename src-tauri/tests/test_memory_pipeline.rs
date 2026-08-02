/// Stage-by-Stage Memory Pipeline Diagnostic Test
/// Tests the EXACT production pipeline used by send_chat_message:
///   1. Memory Extraction (process_user_turn)
///   2. Memory Persistence (SQLite write + verify)
///   3. Memory Retrieval (prepare_injected_messages)
///   4. Prompt Injection (system prompt constructed with memory)
///   5. Cross-Model Persistence (verify after unload/reload)

use std::path::PathBuf;
use sarathi_lib::ai_engine::traits::ChatMessage;
use sarathi_lib::memory_engine::MemoryManager;

fn make_app_data_dir() -> PathBuf {
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| r"C:\Users\lenovo\AppData\Roaming".to_string());
    PathBuf::from(appdata).join("com.sarathi.app")
}

#[tokio::test]
async fn test_memory_pipeline_end_to_end() {
    println!("\n========================================================================");
    println!("   SARATHI P0 MEMORY PIPELINE STAGE-BY-STAGE DIAGNOSTIC                ");
    println!("========================================================================\n");

    let app_data = make_app_data_dir();
    println!("[SETUP] App Data Directory: {:?}", app_data);

    // ========== STAGE 0: Initialize MemoryManager ==========
    println!("\n--- STAGE 0: Initialize MemoryManager ---");
    let mgr = MemoryManager::new(&app_data);
    println!("[STAGE 0] MemoryManager initialized.");
    println!("[STAGE 0] Provider ID: {}", mgr.provider_id());
    println!("[STAGE 0] Active Project ID: {}", mgr.get_active_project_id());

    let health = mgr.check_health().await;
    println!("[STAGE 0] Health Check: {:?}", health);

    let (nodes_before, profile_before, projects_before) = mgr.get_counts().unwrap_or((0, 0, 0));
    println!("[STAGE 0] DB Counts BEFORE: memory_nodes={}, user_profile={}, projects={}", nodes_before, profile_before, projects_before);

    // ========== STAGE 1: Memory Extraction ==========
    println!("\n--- STAGE 1: Memory Extraction ---");
    let user_input = "Hi, my name is Shreyash Patil";
    println!("[STAGE 1] User Input: \"{}\"", user_input);

    let extracted_count = mgr.process_user_turn(user_input, None).await;
    match &extracted_count {
        Ok(count) => println!("[STAGE 1] ✓ Extracted {} fact(s)", count),
        Err(e) => println!("[STAGE 1] ❌ Extraction FAILED: {:?}", e),
    }

    // ========== STAGE 2: Memory Persistence Verification ==========
    println!("\n--- STAGE 2: Memory Persistence Verification ---");
    let (nodes_after, profile_after, _) = mgr.get_counts().unwrap_or((0, 0, 0));
    println!("[STAGE 2] DB Counts AFTER: memory_nodes={} (delta={}), user_profile={} (delta={})",
        nodes_after, nodes_after as isize - nodes_before as isize,
        profile_after, profile_after as isize - profile_before as isize);

    // Check user_profile table
    match mgr.get_user_profile() {
        Ok(profile) => {
            println!("[STAGE 2] User Profile ({} entries):", profile.len());
            for item in &profile {
                println!("   - key='{}', value='{}', category='{}', confidence={}", item.key, item.value, item.category, item.confidence);
            }
            if profile.is_empty() {
                println!("[STAGE 2] ❌ WARNING: No user profile entries found after extraction!");
            }
        }
        Err(e) => println!("[STAGE 2] ❌ Failed to query user_profile: {:?}", e),
    }

    // Search memory_nodes for the fact
    match mgr.search_memories("name", Some("proj_general")).await {
        Ok(memories) => {
            println!("[STAGE 2] Memory Search Results for 'name' ({} results):", memories.len());
            for mem in &memories {
                println!("   - id='{}', type='{}', content='{}', importance={:.2}, similarity={:.2}",
                    mem.id, mem.memory_type, mem.content, mem.importance_score, mem.similarity);
            }
            if memories.is_empty() {
                println!("[STAGE 2] ❌ CRITICAL: No memories returned for query 'name'!");
            }
        }
        Err(e) => println!("[STAGE 2] ❌ Memory search failed: {:?}", e),
    }

    // ========== STAGE 3: Memory Retrieval ==========
    println!("\n--- STAGE 3: Memory Retrieval & Prompt Injection ---");
    let query = "What is my name?";
    println!("[STAGE 3] Query: \"{}\"", query);

    let messages: Vec<ChatMessage> = vec![
        ChatMessage { role: "user".to_string(), content: query.to_string(), timestamp: None },
    ];

    match mgr.prepare_injected_messages(&messages, query).await {
        Ok(injected) => {
            println!("[STAGE 3] ✓ Injected message count: {}", injected.len());
            for (idx, msg) in injected.iter().enumerate() {
                let preview = if msg.content.len() > 400 { &msg.content[..400] } else { &msg.content };
                println!("[STAGE 3] Message[{}] role='{}' content='{}'", idx, msg.role, preview);
            }

            // Verify system prompt contains user name
            let system_msgs: Vec<&ChatMessage> = injected.iter().filter(|m| m.role == "system").collect();
            if system_msgs.is_empty() {
                println!("[STAGE 3] ❌ CRITICAL: No system message injected!");
            } else {
                let sys_content = &system_msgs[0].content;
                let has_name = sys_content.to_lowercase().contains("shreyash") || sys_content.to_lowercase().contains("patil") || sys_content.to_lowercase().contains("name");
                println!("[STAGE 3] System prompt contains user name reference: {}", has_name);
                if !has_name {
                    println!("[STAGE 3] ❌ CRITICAL: System prompt does NOT contain user name!");
                    println!("[STAGE 3] Full system prompt:\n{}", sys_content);
                }
            }
        }
        Err(e) => println!("[STAGE 3] ❌ prepare_injected_messages FAILED: {:?}", e),
    }

    // ========== STAGE 4: Cross-Session Persistence ==========
    println!("\n--- STAGE 4: Cross-Session Persistence (new MemoryManager instance) ---");
    let mgr2 = MemoryManager::new(&app_data);
    println!("[STAGE 4] New MemoryManager created (simulating model switch / app restart)");
    println!("[STAGE 4] Provider ID: {}", mgr2.provider_id());

    match mgr2.get_user_profile() {
        Ok(profile) => {
            println!("[STAGE 4] User Profile ({} entries):", profile.len());
            for item in &profile {
                println!("   - key='{}', value='{}', category='{}'", item.key, item.value, item.category);
            }
        }
        Err(e) => println!("[STAGE 4] ❌ Failed to query profile: {:?}", e),
    }

    match mgr2.search_memories("name", Some("proj_general")).await {
        Ok(memories) => {
            println!("[STAGE 4] Memory Search Results ({} results):", memories.len());
            for mem in &memories {
                println!("   - content='{}', importance={:.2}, similarity={:.2}", mem.content, mem.importance_score, mem.similarity);
            }
            if memories.is_empty() {
                println!("[STAGE 4] ❌ CRITICAL: Memories NOT persisted across MemoryManager instances!");
            } else {
                println!("[STAGE 4] ✓ Memories persisted across MemoryManager instances");
            }
        }
        Err(e) => println!("[STAGE 4] ❌ Search failed: {:?}", e),
    }

    // Verify injection with second MemoryManager
    let messages2: Vec<ChatMessage> = vec![
        ChatMessage { role: "user".to_string(), content: "What is my name?".to_string(), timestamp: None },
    ];
    match mgr2.prepare_injected_messages(&messages2, "What is my name?").await {
        Ok(injected) => {
            println!("[STAGE 4] ✓ Injection with second MemoryManager ({} messages):", injected.len());
            for (idx, msg) in injected.iter().enumerate() {
                let preview = if msg.content.len() > 400 { &msg.content[..400] } else { &msg.content };
                println!("   Message[{}] role='{}' content='{}'", idx, msg.role, preview);
            }
        }
        Err(e) => println!("[STAGE 4] ❌ Injection failed: {:?}", e),
    }

    println!("\n========================================================================");
    println!("   P0 MEMORY PIPELINE DIAGNOSTIC COMPLETE                              ");
    println!("========================================================================\n");
}
