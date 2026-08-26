#!/usr/bin/env node
/**
 * The five acceptance criteria, as one command.
 *
 *   npm run accept
 *
 * PS 26117 names five things a working system must demonstrate. This runs all
 * five and prints one verdict, so "does it meet the brief" has a single answer
 * that anybody can reproduce rather than a conversation.
 *
 *   1. Model auto-selection across at least two task types
 *   2. An agentic task, start to finish
 *   3. A coding task actually run in the sandbox
 *   4. A multimodal task
 *   5. Zero egress in Work mode
 *
 * ## The rule that makes this worth running
 *
 * A criterion that cannot be checked reports **CANNOT VERIFY**, never PASS.
 * Nothing here is allowed to conflate "we could not look" with "it works" —
 * the failure mode that makes a green test suite worse than none at all,
 * because it converts a gap into a reassurance.
 *
 * Criteria that need a provisioned model report BLOCKED, with the exact thing
 * that would unblock them. That is a real state of this system, not an excuse:
 * on a machine with no model weights, no amount of correct code makes the
 * multimodal criterion pass, and pretending otherwise would mislead whoever
 * reads the output.
 */

import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const PASS = 'PASS';
const FAIL = 'FAIL';
const BLOCKED = 'BLOCKED';
const UNVERIFIABLE = 'CANNOT VERIFY';

/**
 * Runs a command and returns both streams whatever the exit code.
 *
 * `spawnSync` rather than `execFileSync` because several of the runners here
 * report on stderr even when they succeed — Python's unittest writes its entire
 * summary there — and a check that could not read the summary would report
 * CANNOT VERIFY for a suite that actually passed.
 */
function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: ROOT,
    encoding: 'utf8',
    timeout: options.timeoutMs ?? 900_000,
    env: { ...process.env, PATH: `${process.env.PATH}${pathSeparator()}${cargoBin()}` },
  });

  return {
    // ENOENT means the toolchain is absent — a different thing from a failing
    // check, and it must never be reported as one.
    absent: result.error?.code === 'ENOENT',
    ok: !result.error && result.status === 0,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
    message: result.error?.message,
  };
}

const pathSeparator = () => (process.platform === 'win32' ? ';' : ':');
const cargoBin = () => join(process.env.HOME ?? process.env.USERPROFILE ?? '', '.cargo', 'bin');

/** Whether a toolchain is present at all, so absence is never read as failure. */
function toolchainPresent(command, args = ['--version']) {
  return run(command, args, { timeoutMs: 60_000 }).ok;
}

/** Counts passing Rust tests matching a filter. */
function cargoTests(filter) {
  if (!toolchainPresent('cargo')) {
    return { verifiable: false, reason: 'cargo is not on PATH, so the Rust tests cannot be run' };
  }

  const result = run('cargo', [
    'test',
    '--manifest-path',
    'src-tauri/Cargo.toml',
    '--lib',
    filter,
    '--',
    '--nocapture',
  ]);

  const summary = /test result: (\w+)\. (\d+) passed; (\d+) failed/.exec(result.stdout);
  if (!summary) {
    return {
      verifiable: false,
      reason: 'the test runner produced no summary line, so nothing can be concluded',
    };
  }

  return {
    verifiable: true,
    passed: Number(summary[2]),
    failed: Number(summary[3]),
  };
}

/**
 * Looks for a provisioned model of a particular kind.
 *
 * `kind` is a predicate over filenames rather than a directory listing, because
 * the distinction that matters here is *what a model can do*, not that some
 * weights exist. A text model on disk says nothing about whether a scan can be
 * read, and treating it as if it did would convert a gap into a reassurance —
 * which is the failure this whole script is written to prevent.
 */
function modelsProvisioned(kind = (f) => f.endsWith('.gguf')) {
  const candidates = [
    join(process.env.APPDATA ?? '', 'com.sarathi.app', 'models'),
    join(process.env.HOME ?? '', '.local', 'share', 'com.sarathi.app', 'models'),
    join(ROOT, 'models'),
  ];

  for (const dir of candidates) {
    if (!dir || !existsSync(dir)) continue;
    try {
      const found = readdirSync(dir, { recursive: true, encoding: 'utf8' });
      if (found.some(kind)) return dir;
    } catch {
      // Unreadable directory is not evidence either way; keep looking.
    }
  }
  return null;
}

/**
 * A vision model, specifically.
 *
 * In llama.cpp a multimodal model is a text model *plus* an `mmproj` projector
 * file — that file is what turns pixels into something the model can attend to.
 * Its presence is the narrowest honest evidence that a scan or a photograph can
 * actually be read on this machine.
 */
const isVisionModel = (file) => {
  const name = file.toLowerCase();
  return name.includes('mmproj') || name.includes('-vision') || name.includes('projector');
};

// ── The five criteria ──────────────────────────────────────────────────

/**
 * 1. Model auto-selection across at least two task types.
 *
 * The router is deterministic and covered by tests that assert a coding request
 * and a document-summary request choose different models *and record different
 * reasons*. Running those tests is a stronger check than driving the UI twice,
 * because it pins the reason string too.
 */
function criterionRouting() {
  const result = cargoTests('registry::router');
  if (!result.verifiable) {
    return { status: UNVERIFIABLE, detail: result.reason };
  }
  if (result.failed > 0) {
    return { status: FAIL, detail: `${result.failed} routing test(s) failed` };
  }
  if (result.passed === 0) {
    return { status: UNVERIFIABLE, detail: 'no routing tests ran, so routing is unproven' };
  }
  return {
    status: PASS,
    detail: `${result.passed} routing checks — different task types select different models, each with a recorded reason`,
  };
}

/**
 * 2. An agentic task, start to finish.
 *
 * Plan → tool calls → evidence → calculation → verified artifact. Covered by
 * the orchestrator and artifact suites, which assert the produced document
 * opens, its citations resolve to passages actually retrieved, and its figures
 * match calculations the engine performed.
 */
function criterionAgenticTask() {
  const orchestrator = cargoTests('orchestrator::');
  const artifacts = cargoTests('artifacts::');

  if (!orchestrator.verifiable || !artifacts.verifiable) {
    return {
      status: UNVERIFIABLE,
      detail: orchestrator.reason ?? artifacts.reason,
    };
  }

  const failed = orchestrator.failed + artifacts.failed;
  if (failed > 0) {
    return { status: FAIL, detail: `${failed} orchestration/artifact test(s) failed` };
  }

  return {
    status: PASS,
    detail: `${orchestrator.passed + artifacts.passed} checks — bounded plan, gated tools, grounded artifact that re-opens and verifies`,
  };
}

/**
 * 3. A coding task actually run in the sandbox.
 *
 * This one is honest about the machine it is on. ARJUN refuses to run
 * model-written code unless a real isolation boundary exists, and on a bare
 * Windows laptop with no container runtime there is none — ARJUN's broker stops
 * ARJUN, but it does not bind a child process. So the criterion reports BLOCKED
 * with the exact thing that would unblock it, rather than claiming a sandbox
 * that is not there.
 */
function criterionSandbox() {
  const result = cargoTests('orchestrator::sandbox');
  if (!result.verifiable) {
    return { status: UNVERIFIABLE, detail: result.reason };
  }
  if (result.failed > 0) {
    return { status: FAIL, detail: `${result.failed} sandbox test(s) failed` };
  }

  const podman = toolchainPresent('podman');
  const docker = toolchainPresent('docker');

  if (!podman && !docker) {
    return {
      status: BLOCKED,
      detail:
        'the refusal path is proven by tests, but no container runtime is installed, so code execution is refused rather than sandboxed. Install Podman to unblock this criterion.',
    };
  }

  return {
    status: PASS,
    detail: `${result.passed} sandbox checks with ${podman ? 'Podman' : 'Docker'} present`,
  };
}

/**
 * 4. A multimodal task.
 *
 * The document pipeline — routing, injection scanning, escalation and
 * confidence flagging — is covered by the sidecar's own tests. The vision
 * engine itself needs a provisioned model, so the criterion reports BLOCKED
 * when none is present.
 */
function criterionMultimodal() {
  if (!toolchainPresent('python')) {
    return {
      status: UNVERIFIABLE,
      detail: 'python is not on PATH, so the document sidecar tests cannot be run',
    };
  }

  const result = run('python', [
    '-m',
    'unittest',
    'discover',
    '-s',
    'sidecars/document_sidecar/tests',
  ]);

  const summary = /Ran (\d+) tests?/.exec(`${result.stderr}${result.stdout}`);
  if (!summary) {
    return { status: UNVERIFIABLE, detail: 'the sidecar test runner produced no summary' };
  }
  if (!result.ok) {
    return { status: FAIL, detail: 'the document sidecar tests failed' };
  }

  const vision = modelsProvisioned(isVisionModel);
  if (!vision) {
    const anyModel = modelsProvisioned();
    return {
      status: BLOCKED,
      detail: `${summary[1]} document-pipeline checks pass — routing, injection scanning, escalation and confidence flagging. But no vision model is provisioned${anyModel ? ' (text weights are present; they cannot read pixels)' : ''}, so scans, handwriting and drawings cannot be read. Download a model with an mmproj projector in Provisioning mode to unblock this.`,
    };
  }

  return {
    status: PASS,
    detail: `${summary[1]} document-pipeline checks, with a vision model provisioned at ${vision}`,
  };
}

/**
 * 5. Zero egress in Work mode.
 *
 * Three independent proofs, all of which must hold: the source gate (no HTTP
 * client outside the broker, no unapproved host), the offline build gate (the
 * build needs no network), and the sovereignty tests (Work mode refuses, the
 * canary fails, mode transitions are audited).
 */
function criterionZeroEgress() {
  const gate = run('node', ['scripts/check-egress.mjs']);
  if (!gate.ok && gate.absent) {
    return { status: UNVERIFIABLE, detail: 'node is not on PATH' };
  }
  if (!gate.ok) {
    return { status: FAIL, detail: 'the source-level egress gate failed — see its output above' };
  }

  const offline = run('node', ['scripts/check-offline-build.mjs']);
  const offlineInconclusive = /INCONCLUSIVE|CANNOT VERIFY/.test(offline.stdout ?? '');
  if (!offline.ok && !offlineInconclusive) {
    return { status: FAIL, detail: 'the build required the network' };
  }

  const sovereignty = cargoTests('sovereignty::');
  if (!sovereignty.verifiable) {
    return { status: UNVERIFIABLE, detail: sovereignty.reason };
  }
  if (sovereignty.failed > 0) {
    return { status: FAIL, detail: `${sovereignty.failed} sovereignty test(s) failed` };
  }

  if (offlineInconclusive) {
    return {
      status: UNVERIFIABLE,
      detail:
        'the source gate and sovereignty tests pass, but a toolchain missing from PATH left the offline build unverified — do not read this as a pass',
    };
  }

  return {
    status: PASS,
    detail: `source gate clean, build needs no network, ${sovereignty.passed} sovereignty checks — Work mode refuses, the canary fails, every decision is audited`,
  };
}

// ── Run them ───────────────────────────────────────────────────────────

const CRITERIA = [
  ['1. Model auto-selection across task types', criterionRouting],
  ['2. Agentic task, start to finish', criterionAgenticTask],
  ['3. Coding task run in the sandbox', criterionSandbox],
  ['4. Multimodal task', criterionMultimodal],
  ['5. Zero egress in Work mode', criterionZeroEgress],
];

const MARK = {
  [PASS]: '  ok  ',
  [FAIL]: ' FAIL ',
  [BLOCKED]: 'BLOCKED',
  [UNVERIFIABLE]: '  ??  ',
};

console.log('\nARJUN acceptance — SIH 2026 PS 26117\n');

const results = [];
for (const [name, check] of CRITERIA) {
  process.stdout.write(`  ${name} … `);
  const outcome = check();
  results.push({ name, ...outcome });
  console.log(`${outcome.status}\n      ${outcome.detail}\n`);
}

const failed = results.filter((r) => r.status === FAIL);
const blocked = results.filter((r) => r.status === BLOCKED);
const unverified = results.filter((r) => r.status === UNVERIFIABLE);
const passed = results.filter((r) => r.status === PASS);

// The baseline record. PS step 15 asks that the scripted run be kept, so it is
// written every time rather than only when somebody remembers to.
const record = {
  // Stamped by the caller's clock — the only clock this script has.
  at: new Date().toISOString(),
  platform: `${process.platform} ${process.arch}`,
  results,
  summary: {
    passed: passed.length,
    failed: failed.length,
    blocked: blocked.length,
    unverified: unverified.length,
  },
};
const recordPath = join(ROOT, 'acceptance-baseline.json');
writeFileSync(recordPath, `${JSON.stringify(record, null, 2)}\n`);

console.log('─'.repeat(72));
console.log(
  `  ${passed.length} pass · ${failed.length} fail · ${blocked.length} blocked · ${unverified.length} unverified`,
);
console.log(`  baseline written to ${recordPath}`);

if (blocked.length > 0) {
  console.log('\n  Blocked criteria need something provisioned, not something fixed:');
  for (const b of blocked) console.log(`    · ${b.name} — ${b.detail}`);
}

if (unverified.length > 0) {
  console.log('\n  Unverified is not a pass. These could not be checked at all:');
  for (const u of unverified) console.log(`    · ${u.name} — ${u.detail}`);
}

console.log('');

// A blocked criterion is a true statement about this machine, so it does not
// fail the run. A failure or an unverifiable check does — silence about what
// could not be checked is the thing this script exists to prevent.
process.exit(failed.length > 0 || unverified.length > 0 ? 1 : 0);
