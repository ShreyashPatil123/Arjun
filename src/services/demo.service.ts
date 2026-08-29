/**
 * Demo scenarios for the SIH presentation layer.
 *
 * Each scenario is a *script* the agent follows: a sequence of synthetic
 * inputs that drive a real end-to-end run, so the judges see ARJUN do its
 * job on realistic (but synthetic — no real MRPL secrets) material.
 *
 * The scenarios are intentionally simple to demo:
 *  - input is in a code constant (not a real file the user must bring)
 *  - the expected output is in a code constant (not a model assertion)
 *  - the run is started via the same `agent.service` the regular
 *    workbench uses, so what the judges see is what an operator sees
 *
 * The service is read-only against the registry. It does not cache,
 * does not poll, and does not touch the network. It only exists to
 * give the demo page typed handles to the three scenarios and to
 * describe what each one will demonstrate, so the page can render
 * the "what this will show" panel without re-deriving it.
 */

export type DemoId = 'pid-analysis' | 'vendor-quote' | 'safety-incident';

/**
 * A scenario the SIH demo page can run.
 *
 * Each one is a *real* run — the demo page sends `prompt` (and
 * optionally `systemPrompt`) to the same `agentService.start` the
 * workbench uses, and renders the standard `RunView` for the result.
 * No fake stepper, no setTimeout: the run surface is identical to
 * the workbench's, which is the whole point.
 */
export interface DemoScenario {
  id: DemoId;
  /** Title shown in the page card. */
  title: string;
  /** One-line description shown under the title. */
  summary: string;
  /**
   * The synthetic input the demo will hand to the agent. Kept short so the
   * judges can read it from the screen.
   */
  prompt: string;
  /**
   * Optional instruction the agent is given in addition to its default
   * system prompt. Used to make the synthetic context unambiguous to the
   * model (e.g. "this is a synthetic scenario, no real plant is at risk").
   */
  systemPrompt?: string;
  /**
   * The skill the agent should reach for. Surfaced on the card so the
   * judge can read the scenario's industrial relevance at a glance.
   */
  skillId?: string;
}

const SCENARIOS: DemoScenario[] = [
  {
    id: 'pid-analysis',
    title: 'P&ID Analysis',
    summary:
      'Synthetic P&ID, synthetic equipment register, real skill chain. ' +
      'ARJUN reads the drawing, identifies the equipment, cross-references ' +
      'with the register, and drafts an inspection note.',
    prompt:
      'Synthetic P&ID A-101-001 Rev 6 attached. ' +
      'Identify the equipment on the drawing, cross-reference V-101 against ' +
      'the equipment register, and draft an inspection note for the next ' +
      'quarterly survey.',
    systemPrompt:
      'This is a SIH 2026 demo on synthetic data. No real plant is at risk. ' +
      'The P&ID A-101-001 Rev 6 is generated for the demo and lives in the ' +
      'demo fixtures directory. Be honest about what you read and what you ' +
      'do not read; do not invent tags or values that are not on the page.',
    skillId: 'pid-reader',
  },
  {
    id: 'vendor-quote',
    title: 'Vendor Quote Review',
    summary:
      'Two synthetic quotes, a real comparison framework, and a real 3-year ' +
      'TCO. ARJUN surfaces risk flags the procurement committee should see.',
    prompt:
      'Synthetic vendor quotes for a 75 kW centrifugal pump attached ' +
      '(Quote A and Quote B). Compare against the standard procurement ' +
      'template, flag the unusual terms, and produce an approval memo.',
    systemPrompt:
      'This is a SIH 2026 demo on synthetic data. The quotes are illustrative, ' +
      'not from any real vendor. Use the vendor-evaluator skill; do not ' +
      'invent clauses or figures the documents do not support.',
    skillId: 'vendor-evaluator',
  },
  {
    id: 'safety-incident',
    title: 'Safety Incident Analysis',
    summary:
      'A synthetic incident, real SOPs, real safety clauses. ARJUN finds the ' +
      'deviation, identifies root cause, and drafts a corrective action plan.',
    prompt:
      'Synthetic incident: low-low level alarm on V-101 at 02:14. Operator ' +
      'found level at 6%, pump P-101B discharge pressure at 4.2 bar (design ' +
      '12 bar), and ESD-0101 had not actuated. Investigate.',
    systemPrompt:
      'This is a SIH 2026 demo on synthetic data. The event itself is ' +
      'generated for the demo; the SOP references and ESD matrix are ' +
      'real. Use the safety-compliance and equipment-lookup skills. Cite ' +
      'every claim. Do not invent findings the documents do not support.',
    skillId: 'safety-compliance',
  },
];

export const demoService = {
  list(): DemoScenario[] {
    return SCENARIOS;
  },
  get(id: DemoId): DemoScenario | undefined {
    return SCENARIOS.find((s) => s.id === id);
  },
};
