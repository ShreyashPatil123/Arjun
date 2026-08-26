#!/usr/bin/env node
/**
 * Proves the bundle gate fails when it should.
 *
 * A gate nobody has watched fail is a gate nobody knows works — and this one is
 * the strongest piece of sovereignty evidence the project has, so "it passes"
 * is not enough to know about it. Each case here doctors a copy of the real
 * bundle to reintroduce exactly one thing the gate exists to catch, and asserts
 * the gate says so.
 *
 * Working on a copy matters: the shipped artifact is never modified, so a failed
 * run here cannot leave a tampered bundle behind for the next command to use.
 */

import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const GATE = join(ROOT, 'scripts', 'check-bundle.mjs');
const BUNDLE = join(ROOT, 'agent-runtime', 'dist', 'arjun-agent-runtime.mjs');

if (!existsSync(BUNDLE)) {
  console.error('bundle gate self-test: the bundle is missing. Run `npm run runtime:build` first.');
  process.exit(2);
}

const work = mkdtempSync(join(tmpdir(), 'arjun-bundle-gate-'));
const failures = [];

/** Runs the gate over one doctored copy and returns what it said. */
function runGate(name, doctor) {
  const target = join(work, `${name}.mjs`);
  copyFileSync(BUNDLE, target);
  const doctored = doctor(readFileSync(target, 'utf8'));
  writeFileSync(target, doctored);
  const result = spawnSync(process.execPath, [GATE, target], { encoding: 'utf8' });
  return { code: result.status, output: `${result.stdout}${result.stderr}` };
}

function expectFailure(name, expectedText, doctor) {
  const { code, output } = runGate(name, doctor);
  if (code === 0) {
    failures.push(`${name}: the gate PASSED a bundle it should have refused`);
    return;
  }
  if (!output.includes(expectedText)) {
    failures.push(
      `${name}: the gate failed, but for the wrong reason.\n` +
        `    expected to see: ${expectedText}\n    got: ${output.trim().split('\n')[1] ?? output.trim()}`,
    );
    return;
  }
  console.log(`  ${name} … caught`);
}

console.log('bundle gate self-test — each case reintroduces one thing the gate must catch\n');

// The gate's whole purpose: a second protocol adapter would mean a vendor API
// is reachable from the loop.
expectFailure('a-second-adapter', 'extra protocol adapter', (source) =>
  source.replace(
    'createLazyRegistration(\n    "openai-completions"',
    'createLazyRegistration(\n    "anthropic-messages"',
  ).replace('createLazyRegistration("openai-completions"', 'createLazyRegistration("anthropic-messages"'),
);

// A channel is a way for a document to leave the plant.
expectFailure('a-channel', 'excluded capability', (source) =>
  `${source}\nconst sendTo = "https://api.telegram.org/bot";\n`,
);

// The cloud code-execution tool the reuse blueprint singles out.
expectFailure('cloud-code-execution', 'excluded capability', (source) =>
  `${source}\nconst tool = { name: "code_execution" };\n`,
);

// Absence checks pass on an empty file; the defences have to be asserted too.
expectFailure('missing-loopback-guard', 'missing defence', (source) =>
  source.replaceAll('is not loopback', 'is fine actually'),
);

expectFailure('missing-grant-check', 'missing defence', (source) =>
  source.replaceAll('No authorisation grant for', 'Proceeding without checking'),
);

// A host nobody has reviewed appearing in the artifact.
expectFailure('an-unreviewed-host', 'unreviewed host', (source) =>
  `${source}\nconst endpoint = "https://api.anthropic.com/v1/messages";\n`,
);

// More of a reviewed host than was reviewed — something new is carrying it.
expectFailure('a-grown-host-surface', 'host surface grew', (source) =>
  `${source}\n// https://api.together.xyz https://api.together.xyz https://api.together.xyz\n`,
);

// If the bundler changes how registrations are emitted, the matcher stops
// matching and the gate silently stops checking. That must fail loudly.
expectFailure('a-blind-gate', 'gate integrity', (source) =>
  source.replaceAll('createLazyRegistration(', 'createRegistrationRenamed('),
);

rmSync(work, { recursive: true, force: true });

if (failures.length > 0) {
  console.error(`\nbundle gate self-test: FAIL — ${failures.length} case(s)\n`);
  for (const failure of failures) console.error(`  ${failure}`);
  console.error('\nThe gate is not catching what it claims to. Fix it before trusting a pass.');
  process.exit(1);
}

console.log('\nbundle gate self-test: pass — the gate refuses every regression it claims to catch');
