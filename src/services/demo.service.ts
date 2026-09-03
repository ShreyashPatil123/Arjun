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
 *
 * ## The documents are real files
 *
 * Every scenario's prompt says its documents are "attached". Until
 * `DemoLaunch` existed, nothing was: the page dispatched a prompt, a title and
 * a framing string, and no documents at all. The model was asked to
 * cross-reference a drawing it had never been given — and, being asked about
 * the organisation's record with nothing retrieved, either refused or invented.
 * Both were shown to a judge as the product working.
 *
 * The fixtures are checked in under `src/demo-fixtures/`, in plain text, so
 * a reviewer can open the directory and read exactly what the model was handed.
 * Each one states what it does *not* contain, because what the model does when
 * a document does not answer the question is the part worth watching.
 */
import type { Classification, ComposerAttachment } from './agent.service';

import equipmentRegister from '../demo-fixtures/equipment-register.txt?raw';
import incidentReport from '../demo-fixtures/incident-report.txt?raw';
import pidA101001 from '../demo-fixtures/pid-A-101-001.txt?raw';
import vendorQuotes from '../demo-fixtures/vendor-quotes.txt?raw';

export type DemoId = 'pid-analysis' | 'vendor-quote' | 'safety-incident';

/**
 * A scenario the SIH demo page can run.
 *
 * Each one is a *real* run — the demo page sends `prompt` (and
 * optionally `scenarioInstructions`) to the same `agentService.start` the
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
  /**
   * Extra framing for this scenario, appended *beneath* ARJUN's own
   * instructions rather than replacing them.
   *
   * It used to be `systemPrompt` and it replaced the core, so a scenario could
   * remove the retrieval rule, the citation rule and the honesty rule — and
   * the demo would look normal while answering from the model's weights.
   */
  scenarioInstructions?: string;
  /**
   * The skill the agent should reach for. Surfaced on the card so the
   * judge can read the scenario's industrial relevance at a glance.
   */
  skillId?: string;
  /**
   * The checked-in synthetic documents this scenario hands to the run.
   *
   * File names under `src/demo-fixtures/`. Every scenario's prompt used to
   * say "attached" while nothing was attached: the demo dispatched a prompt, a
   * title and a framing string and no documents at all. The model was asked to
   * cross-reference a drawing it had never been given, and — being asked
   * about the organisation's record with nothing retrieved — either refused or
   * invented, and both were shown to a judge as the product working.
   */
  fixtures: string[];
  /**
   * The sensitivity the run is started at.
   *
   * Sent rather than left to default, because it decides which models may see
   * the material and the demo should be showing that decision, not bypassing
   * it.
   */
  classification: Classification;
}

/**
 * What a demo launch actually sends.
 *
 * Typed so that a scenario cannot claim an input it does not carry. The three
 * fields below `prompt` are the ones that were missing entirely: the documents,
 * the scenario's own identity, and the skill the card advertises.
 */
export interface DemoLaunch {
  /** Which scenario this is, carried through so a run can be traced to it. */
  scenarioId: DemoId;
  prompt: string;
  classification: Classification;
  /** Appended beneath ARJUN's own instructions; never replaces them. */
  scenarioInstructions?: string;
  /** The skill this scenario asks the run to load. */
  skillId?: string;
  /** The documents, read from `src/demo-fixtures/` and carried as bytes. */
  attachments: ComposerAttachment[];
}

const SCENARIOS: DemoScenario[] = [
  {
    id: 'pid-analysis',
    title: 'P&ID Analysis',
    summary:
      'A synthetic P&ID and a synthetic equipment register, both checked in ' +
      'and handed to the run. ARJUN is asked to read the drawing, ' +
      'cross-reference V-101 against the register, and draft an inspection ' +
      'note. What it actually does is on the trace.',
    prompt:
      'The synthetic P&ID A-101-001 Rev 6 and the equipment register are attached. ' +
      'Identify the equipment on the drawing, cross-reference V-101 against ' +
      'the equipment register, and draft an inspection note for the next ' +
      'quarterly survey.',
    scenarioInstructions:
      'This is a SIH 2026 demo on synthetic data. No real plant is at risk. ' +
      'The P&ID A-101-001 Rev 6 and the equipment register are generated for ' +
      'the demo and are attached to this turn. Be honest about what you read ' +
      'and what you do not read; do not invent tags or values that are not on ' +
      'the page.',
    skillId: 'pid-reader',
    fixtures: ['pid-A-101-001.txt', 'equipment-register.txt'],
    classification: 'processDiagram',
  },
  {
    id: 'vendor-quote',
    title: 'Vendor Quote Review',
    summary:
      'Two synthetic quotations for one duty, checked in and handed to the ' +
      'run. ARJUN is asked to compare them and flag the unusual terms. ' +
      'Neither quote states an energy figure or a maintenance cost, so a ' +
      'lifetime-cost claim would be invented — which is part of what this ' +
      'shows.',
    prompt:
      'Two synthetic vendor quotes for a 75 kW centrifugal pump are attached ' +
      '(Quote A and Quote B). Compare against the standard procurement ' +
      'template, flag the unusual terms, and produce an approval memo.',
    scenarioInstructions:
      'This is a SIH 2026 demo on synthetic data. The quotes are illustrative, ' +
      'not from any real vendor. Use the vendor-evaluator skill; do not ' +
      'invent clauses or figures the documents do not support.',
    skillId: 'vendor-evaluator',
    fixtures: ['vendor-quotes.txt'],
    classification: 'internal',
  },
  {
    id: 'safety-incident',
    title: 'Safety Incident Analysis',
    summary:
      'A synthetic shift log, with the P&ID and register it refers to. ARJUN ' +
      'is asked to investigate. The log deliberately omits the ESD logic and ' +
      'the alarm set point, so an answer that states a root cause without ' +
      'citing a document has invented it.',
    prompt:
      'Synthetic incident: low-low level alarm on V-101 at 02:14. Operator ' +
      'found level at 6%, pump P-101B discharge pressure at 4.2 bar (design ' +
      '12 bar), and ESD-0101 had not actuated. Investigate.',
    scenarioInstructions:
      'This is a SIH 2026 demo on synthetic data. The event itself is ' +
      'generated for the demo, and so are the documents attached to it. The ' +
      'shift log does not contain the ESD logic or the alarm set point. Cite ' +
      'every claim against the document it came from, and say plainly when ' +
      'the attached documents do not answer the question.',
    skillId: 'safety-compliance',
    fixtures: ['incident-report.txt', 'pid-A-101-001.txt', 'equipment-register.txt'],
    classification: 'internal',
  },
];

/**
 * The checked-in documents, by file name.
 *
 * Inlined at build time with `?raw` rather than fetched at runtime. Two
 * reasons, and the second is the one that matters:
 *
 *  - A `fetch` — even of a same-origin bundled asset — constructs an HTTP
 *    client outside the sovereignty broker, which is exactly the shape
 *    `scripts/check-egress.mjs` exists to catch. It caught this one. Silencing
 *    it with an `arjun-egress-ok` annotation would have spent a real gate's
 *    credibility on a call that never needed to be a network call at all.
 *  - A fixture that goes missing is now a *build* failure. Vite cannot resolve
 *    the import, so the demo cannot ship without its documents rather than
 *    discovering it in front of a judge.
 *
 * The files stay checked in as plain text under `src/demo-fixtures/`, so a
 * reviewer can still open the directory and read exactly what the model was
 * handed.
 */
const FIXTURES: Record<string, string> = {
  'equipment-register.txt': equipmentRegister,
  'incident-report.txt': incidentReport,
  'pid-A-101-001.txt': pidA101001,
  'vendor-quotes.txt': vendorQuotes,
};

/**
 * Reads one checked-in fixture into something that can cross the Tauri
 * boundary.
 *
 * An unknown or empty fixture throws rather than resolving to an empty
 * attachment — a demo that quietly ran with no documents is the failure this
 * whole change exists to remove.
 */
async function readFixture(name: string): Promise<ComposerAttachment> {
  const text = FIXTURES[name];
  if (text === undefined) {
    throw new Error(
      `The demo fixture ${name} is not one of the checked-in documents. The scenario was ` +
        'not started, because running it without its documents would ask the model to ' +
        'cross-reference something it had never been given.',
    );
  }
  if (text.trim().length === 0) {
    throw new Error(`The demo fixture ${name} is empty, so the scenario was not started.`);
  }
  return {
    name,
    mime: 'text/plain',
    // `btoa` handles the Latin-1 range; these fixtures are ASCII by
    // construction, and a non-ASCII byte would throw here rather than arrive
    // silently mangled.
    dataBase64: btoa(text),
  };
}

export const demoService = {
  list(): DemoScenario[] {
    return SCENARIOS;
  },
  get(id: DemoId): DemoScenario | undefined {
    return SCENARIOS.find((s) => s.id === id);
  },

  /**
   * Builds everything a scenario needs to be run for real.
   *
   * Rejects rather than degrading. A scenario whose documents cannot be read is
   * a scenario that must not start: its prompt says "attached", and a run
   * begun without them would be answering about a drawing it never saw.
   */
  async launch(id: DemoId): Promise<DemoLaunch> {
    const scenario = SCENARIOS.find((s) => s.id === id);
    if (!scenario) throw new Error(`There is no demo scenario called ${id}.`);

    const attachments = await Promise.all(scenario.fixtures.map(readFixture));

    return {
      scenarioId: scenario.id,
      prompt: scenario.prompt,
      classification: scenario.classification,
      scenarioInstructions: scenario.scenarioInstructions,
      skillId: scenario.skillId,
      attachments,
    };
  },
};
