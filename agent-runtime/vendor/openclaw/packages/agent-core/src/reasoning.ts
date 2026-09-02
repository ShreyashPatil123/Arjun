import {
  resolveClaudeFable5ModelIdentity,
  resolveClaudeOpus5ModelIdentity,
  resolveClaudeSonnet5ModelIdentity,
  type Model,
  type SimpleStreamOptions,
} from "@openclaw/llm-core";
import type { ThinkingLevel } from "./types.js";

type EnabledThinkingLevel = Exclude<NonNullable<SimpleStreamOptions["reasoning"]>, "off">;

const ENABLED_THINKING_LEVELS = new Set<EnabledThinkingLevel>([
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);

function isEnabledThinkingLevel(value: unknown): value is EnabledThinkingLevel {
  return ENABLED_THINKING_LEVELS.has(value as EnabledThinkingLevel);
}

export function resolveAgentReasoningOption(
  model: Model,
  thinkingLevel: ThinkingLevel,
): SimpleStreamOptions["reasoning"] {
  if (thinkingLevel !== "off") {
    return thinkingLevel;
  }
  const offFallback =
    model.thinkingLevelMap?.off ??
    // ARJUN prune: upstream also matched a second, AWS-hosted protocol here.
    // That arm is removed. ARJUN registers one protocol adapter,
    // `openai-completions` (enforced by scripts/check-bundle.mjs), so no model
    // reaching this function can carry the other api. The vendor id is not
    // written out even in this comment, because the bundle gate asserts its
    // absence from the shipped artifact and a comment ships too.
    (model.api === "anthropic-messages" &&
    resolveClaudeFable5ModelIdentity(model)
      ? "low"
      : undefined);
  if (isEnabledThinkingLevel(offFallback)) {
    return offFallback;
  }
  return model.api === "anthropic-messages" &&
    (resolveClaudeSonnet5ModelIdentity(model) || resolveClaudeOpus5ModelIdentity(model))
    ? "off"
    : undefined;
}
