//! Decoupled Intent Detection Engine
//!
//! Classifies user prompts into high-level intent categories:
//! User Prompt -> Intent Classifier -> Target Capability -> Compatible Adapter Selection.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptIntent {
    Coding,
    Reasoning,
    Mathematics,
    ToolCalling,
    Research,
    GeneralChat,
}

impl PromptIntent {
    pub fn to_capability_name(&self) -> &'static str {
        match self {
            Self::Coding => "coding",
            Self::Reasoning => "reasoning",
            Self::Mathematics => "mathematics",
            Self::ToolCalling => "tool-calling",
            Self::Research => "research",
            Self::GeneralChat => "general",
        }
    }
}

pub struct IntentDetector;

impl IntentDetector {
    /// Classifies user prompt into a `PromptIntent`
    pub fn classify(prompt: &str) -> PromptIntent {
        let lower = prompt.to_lowercase();

        // 1. Tool Execution Intent
        if lower.contains("json")
            || lower.contains("api")
            || lower.contains("function call")
            || lower.contains("arguments")
            || lower.contains("schema")
            || lower.contains("execute tool")
        {
            return PromptIntent::ToolCalling;
        }

        // 2. Code Generation & Debugging Intent
        if lower.contains("code")
            || lower.contains("function ")
            || lower.contains("def ")
            || lower.contains("fn ")
            || lower.contains("class ")
            || lower.contains("import ")
            || lower.contains("python")
            || lower.contains("rust")
            || lower.contains("javascript")
            || lower.contains("typescript")
            || lower.contains("bug")
            || lower.contains("refactor")
            || lower.contains("error")
            || lower.contains("exception")
            || lower.contains("html")
            || lower.contains("css")
            || lower.contains("sql")
        {
            return PromptIntent::Coding;
        }

        // 3. Mathematics Intent
        if lower.contains("calculate")
            || lower.contains("equation")
            || lower.contains("integral")
            || lower.contains("derivative")
            || lower.contains("matrix")
            || lower.contains("theorem")
            || lower.contains("proof")
            || lower.contains("algebra")
            || lower.contains("geometry")
            || lower.contains("solve for")
            || lower.contains("math")
            || lower.contains(" + ")
            || lower.contains(" = ")
        {
            return PromptIntent::Mathematics;
        }

        // 4. Step-by-Step Reasoning Intent
        if lower.contains("think step by step")
            || lower.contains("reason")
            || lower.contains("explain why")
            || lower.contains("tradeoff")
            || lower.contains("pros and cons")
            || lower.contains("compare")
            || lower.contains("logical")
            || lower.contains("deduce")
        {
            return PromptIntent::Reasoning;
        }

        // 5. Research & Document Analysis Intent
        if lower.contains("summarize")
            || lower.contains("key takeaways")
            || lower.contains("literature")
            || lower.contains("paper")
            || lower.contains("citation")
            || lower.contains("findings")
        {
            return PromptIntent::Research;
        }

        PromptIntent::GeneralChat
    }
}
