/**
 * Tools the model may request, and the single point at which that request is
 * authorised.
 *
 * ## Where the boundary is
 *
 * Nothing in this file decides anything. Every tool here is a stub that forwards
 * to the Rust core, and every call is put through `authorizeToolCall` first,
 * which asks the core for a verdict from `orchestrator::gateway::ToolGateway`.
 * The model can request; only Rust decides. That split is the product's central
 * claim, so it is worth being precise about how it is held:
 *
 * 1. **Authorisation is a loop hook, not a tool method.** It runs in
 *    `beforeToolCall`, which agent-core applies to every call uniformly. A tool
 *    added in a later phase cannot forget to authorise itself, because it is not
 *    the tool's job.
 *
 * 2. **A verdict is a single-use grant, not a boolean.** Rust replies to an
 *    allow with an opaque token bound to that exact call and consumes it on
 *    execution. So this side cannot cache a verdict, replay one, or authorise
 *    cheap arguments and execute expensive ones -- not because it is careful,
 *    but because the token would not match. The check is structural.
 *
 * 3. **Rust re-checks anyway.** `tool.execute` validates independently of the
 *    grant. Two independent refusals beat one, and the grant protects against a
 *    compromised runtime while the re-check protects against a bug in the grant.
 */

import { Type } from "typebox";
import type { AgentTool, BeforeToolCallContext, BeforeToolCallResult } from "@openclaw/agent-core";
import { ErrorCode, type ErrorCodeValue } from "./protocol.js";
import { RpcError, type RpcPeer } from "./peer.js";

/** What the Rust gateway replies to `tool.authorize`. Mirrors `GatewayVerdict`. */
export type Verdict =
  | { outcome: "allow"; tool: string; grant: string; resolvedPath?: string | null }
  | { outcome: "needsApproval"; tool: string; summary: string; resolvedPath?: string | null }
  | { outcome: "refuse"; reason: string };

/** What Rust returns from `tool.execute`. */
export interface ToolExecution {
  /** What the model sees. */
  text: string;
  /** Structured detail for the audit record and the UI. Never shown to the model. */
  details?: unknown;
}

/**
 * Grants held between authorisation and execution, keyed by tool-call id.
 *
 * Scoped per run and cleared when it ends, so a grant cannot outlive the run
 * that earned it even if Rust's own expiry were to fail.
 */
export class GrantLedger {
  readonly #grants = new Map<string, string>();

  put(toolCallId: string, grant: string): void {
    this.#grants.set(toolCallId, grant);
  }

  /** Reads and removes. A grant is good for exactly one execution. */
  take(toolCallId: string): string | undefined {
    const grant = this.#grants.get(toolCallId);
    this.#grants.delete(toolCallId);
    return grant;
  }

  clear(): void {
    this.#grants.clear();
  }

  get size(): number {
    return this.#grants.size;
  }
}

/**
 * Asks Rust whether a call may proceed, and records the grant if it may.
 *
 * Returns a `BeforeToolCallResult` for agent-core: `{ block: true, reason }`
 * turns into an error tool result the model reads and can recover from, which is
 * the behaviour we want -- a refusal is information, not a crash.
 */
export async function authorizeToolCall(
  peer: RpcPeer,
  ledger: GrantLedger,
  runId: string,
  context: BeforeToolCallContext,
): Promise<BeforeToolCallResult | undefined> {
  const { toolCall, args } = context;
  let verdict: Verdict;
  try {
    verdict = (await peer.request("tool.authorize", {
      runId,
      toolCallId: toolCall.id,
      tool: toolCall.name,
      args,
    })) as Verdict;
  } catch (error) {
    // A gateway that cannot be reached is a gateway that did not say yes.
    // Failing closed is the only safe reading of silence here.
    const message = error instanceof Error ? error.message : String(error);
    return { block: true, reason: `Tool authorisation is unavailable, so the call was not made: ${message}` };
  }

  switch (verdict.outcome) {
    case "allow":
      ledger.put(toolCall.id, verdict.grant);
      return undefined;
    case "needsApproval":
      // Phase 1 ships only tools the gateway marks `needs_approval: false`, so
      // this is unreachable today. It blocks rather than assuming consent
      // because the wrong default here is the one that cannot be undone; the
      // approval queue is wired in Phase 4.
      return {
        block: true,
        reason: `${verdict.summary}\n\nThis action needs a person to approve it, and approval is not yet wired into this runtime.`,
      };
    case "refuse":
      return { block: true, reason: verdict.reason };
  }
}

/** Builds one tool whose execution is performed by the Rust core. */
function hostTool<TSchema extends ReturnType<typeof Type.Object>>(options: {
  name: string;
  label: string;
  description: string;
  parameters: TSchema;
  peer: RpcPeer;
  ledger: GrantLedger;
  runId: string;
  modelId: string;
  /**
   * Whether this tool may run alongside others in the same turn.
   *
   * Read-only tools are parallel: several searches at once cost the operator
   * the slowest rather than the sum, and one search cannot affect what another
   * returns. Anything that writes, produces a file or runs code is sequential —
   * two writes to the same path in one turn have an order, and it should not be
   * whichever finished first.
   */
  executionMode: "parallel" | "sequential";
}): AgentTool {
  const { name, label, description, parameters, peer, ledger, runId, modelId, executionMode } =
    options;
  return {
    name,
    label,
    description,
    parameters,
    executionMode,
    async execute(toolCallId, params) {
      const grant = ledger.take(toolCallId);
      if (!grant) {
        // Reached only if the loop skipped `beforeToolCall` or a grant was
        // consumed twice. Either is a defect in this runtime, and the honest
        // response is to refuse and say so rather than try the call anyway.
        throw new RpcError(
          ErrorCode.Refused,
          `No authorisation grant for ${name}. The call was not put through the gateway.`,
        );
      }
      const execution = (await peer.request("tool.execute", {
        runId,
        toolCallId,
        tool: name,
        args: params,
        grant,
        // Stamped onto anything this call produces, so a reader of the
        // document knows which model wrote it.
        model: modelId,
      })) as ToolExecution;
      return {
        content: [{ type: "text", text: execution.text }],
        details: execution.details ?? null,
      };
    },
  } as AgentTool;
}

/**
 * The catalogue.
 *
 * Mirrors `ToolName` in `src-tauri/src/orchestrator/tools.rs`, which is the
 * authority — a name absent there is refused by the gateway regardless of what
 * is declared here, and a name present there but missing here simply cannot be
 * asked for.
 *
 * The descriptions are load-bearing. A local 7B model decides which tool to
 * reach for from these sentences alone, and the common failures are all
 * addressable in prose: answering from memory instead of searching, writing an
 * absolute path, recomputing a figure the calculation engine already produced,
 * describing a document it never created. Each description says the thing that
 * prevents its own tool's characteristic mistake.
 */
export function buildTools(
  peer: RpcPeer,
  ledger: GrantLedger,
  runId: string,
  modelId: string,
): AgentTool[] {
  const shared = { peer, ledger, runId, modelId };
  return [
    hostTool({
      ...shared,
      name: "search_documents",
      label: "Search documents",
      description:
        "Search the organisation's own indexed documents and return passages with their source " +
        "document and page. Results are filtered by what the signed-in user is permitted to read, " +
        "so an absent document may exist but be out of scope rather than not exist. Use this " +
        "before answering anything about internal procedure, specification, or correspondence; do " +
        "not answer such questions from memory.",
      parameters: Type.Object({
        query: Type.String({
          description:
            "What to look for, in natural language. Specific technical terms retrieve better than paraphrase.",
        }),
      }),
      executionMode: "parallel",
    }),

    hostTool({
      ...shared,
      name: "read_scoped_file",
      label: "Read a file",
      description:
        "Read a text file from this task's working directory. Only that directory is readable; " +
        "any other path is refused. Use a relative name such as \"draft.md\". Binary files and " +
        "documents are not read this way — use the document tools for those.",
      parameters: Type.Object({
        path: Type.String({ description: "Relative to the task's working directory." }),
      }),
      executionMode: "parallel",
    }),

    hostTool({
      ...shared,
      name: "run_calculation",
      label: "Calculate",
      description:
        "Evaluate an arithmetic expression with units, deterministically, showing every step. " +
        "Use this for any number that will appear in a deliverable — a figure you worked out in " +
        "your head is not verifiable and may be wrong. Quote the result exactly as returned; do " +
        "not round it again or recompute it.",
      parameters: Type.Object({
        expression: Type.String({
          description: 'With units, for example "1500 kg / 3 m^3" or "0.85 * 240 kW".',
        }),
      }),
      executionMode: "parallel",
    }),

    hostTool({
      ...shared,
      name: "validate_artifact",
      label: "Check a produced file",
      description:
        "Re-open a file this task produced and confirm it is sound. Use it after producing a " +
        "document or workbook, before telling anyone it is ready.",
      parameters: Type.Object({
        path: Type.String({ description: "Relative to the task's working directory." }),
      }),
      executionMode: "parallel",
    }),

    hostTool({
      ...shared,
      name: "write_scoped_file",
      label: "Write a file",
      description:
        "Write a text file into this task's working directory. A person must approve it before " +
        "it happens, so expect a pause. Use this for notes and drafts; use create_docx or " +
        "create_xlsx for deliverables somebody will be handed.",
      parameters: Type.Object({
        path: Type.String({ description: "Relative to the task's working directory." }),
        content: Type.String({ description: "The complete file contents." }),
      }),
      executionMode: "sequential",
    }),

    hostTool({
      ...shared,
      name: "create_docx",
      label: "Produce a Word document",
      description:
        "Produce a Word document from a template. A person must approve it before it happens. " +
        "Supply every field the template asks for as text; a missing required field fails the " +
        "render rather than producing a document with a gap in it. The result is marked DRAFT " +
        "until somebody signs it. Available templates: approval_note.",
      parameters: Type.Object({
        path: Type.String({ description: 'Relative, for example "approval-note.docx".' }),
        template: Type.String({ description: "Template name, for example approval_note." }),
        content: Type.Record(Type.String(), Type.String(), {
          description: "Field name to text. Every value must be a string.",
        }),
      }),
      executionMode: "sequential",
    }),

    hostTool({
      ...shared,
      name: "create_xlsx",
      label: "Produce a calculation workbook",
      description:
        "Produce an Excel workbook showing the working for every calculation this task has run, " +
        "as live formulas Excel recomputes. A person must approve it. It draws on the " +
        "run_calculation calls already made — it does not take figures as arguments, so run the " +
        "calculations first.",
      parameters: Type.Object({
        path: Type.String({ description: 'Relative, for example "working.xlsx".' }),
      }),
      executionMode: "sequential",
    }),

    hostTool({
      ...shared,
      name: "execute_code",
      label: "Run code in the sandbox",
      description:
        "Run code in an isolated sandbox with no network access. A person must approve it. " +
        "NOTE: execution is not built yet — this call is accepted, checked and then refused, and " +
        "nothing runs. Treat a refusal as final: say the code was not run rather than describing " +
        "what it would have produced. Use run_calculation for arithmetic, which does work.",
      parameters: Type.Object({
        language: Type.String({ description: "For example python." }),
        source: Type.String({ description: "The complete program." }),
      }),
      executionMode: "sequential",
    }),
  ];
}

export const toolErrorCode: ErrorCodeValue = ErrorCode.ToolFailed;
