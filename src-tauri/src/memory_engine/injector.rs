//! Prompt Injection Engine
//! Constructs user profile + working memory + project memory + retrieved memories
//! and injects them into the system prompt BEFORE every model inference turn.

use crate::ai_engine::traits::ChatMessage;
use crate::memory_engine::traits::ScoredCandidate;

pub struct PromptInjector;

impl PromptInjector {
    /// Injects Sarathi's unified memory into system prompt messages
    pub fn inject_memory_into_messages(
        messages: &[ChatMessage],
        user_profile_summary: &str,
        project_name: &str,
        retrieved_memories: &[ScoredCandidate],
        rolling_summary: Option<&str>,
    ) -> Vec<ChatMessage> {
        let start_time = std::time::Instant::now();
        let mut memory_section = String::new();

        memory_section.push_str(&format!("User Workspace & Project Context: {}\n", project_name));

        let has_profile = !user_profile_summary.is_empty() && !user_profile_summary.contains("No prior user preferences recorded.");
        if has_profile {
            memory_section.push_str("\nKnown User Information & Preferences:\n");
            memory_section.push_str(user_profile_summary);
            memory_section.push('\n');
        }

        if !retrieved_memories.is_empty() {
            memory_section.push_str("\nRecalled Context & Facts:\n");
            for (idx, mem) in retrieved_memories.iter().enumerate() {
                memory_section.push_str(&format!("{}. {}\n", idx + 1, mem.content));
            }
        }

        if let Some(summary) = rolling_summary {
            if !summary.is_empty() {
                memory_section.push_str("\nConversation History Summary:\n");
                memory_section.push_str(summary);
                memory_section.push('\n');
            }
        }

        let mut final_messages = Vec::new();
        let mut has_system = false;

        for msg in messages {
            if msg.role == "system" {
                has_system = true;
                let combined_content = format!("{}\n\n{}", memory_section, msg.content);
                final_messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: combined_content,
                    timestamp: msg.timestamp.clone(),
                });
            } else {
                final_messages.push(msg.clone());
            }
        }

        if !has_system {
            let memory_directive = "\nInstructions:\n- You are Sarathi, an intelligent local AI companion.\n- Use the Known User Information & Preferences and Recalled Context above to personalize your responses.\n- When the user asks about themselves, their name, preferences, or past context, directly answer using the stored user information provided above.";
            final_messages.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: format!("{}\n{}", memory_section, memory_directive),
                    timestamp: None,
                },
            );
        }

        let latency_ms = start_time.elapsed().as_millis();
        log::info!(
            "[PROMPT_INJECTION] Successfully injected Sarathi Memory into prompt in {}ms | Project: '{}' | Has Profile: {} | Recalled Memories: {} | Total Messages: {}",
            latency_ms, project_name, has_profile, retrieved_memories.len(), final_messages.len()
        );

        if let Some(sys_msg) = final_messages.iter().find(|m| m.role == "system") {
            log::info!(
                "[PROMPT_INJECTION_SYSTEM_PAYLOAD] Final System Prompt ({}) bytes):\n{}",
                sys_msg.content.len(), sys_msg.content
            );
        }

        final_messages
    }
}
