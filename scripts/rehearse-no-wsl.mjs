#!/usr/bin/env node
/**
 * The WSL2-disabled rehearsal.
 *
 *   npm run rehearse
 *
 * The stated risk in the build plan is blunt: *nothing on the critical path may
 * require WSL2, because it is the most likely thing to be missing or broken on
 * demo day.* A claim like that is worth exactly as much as the last time
 * somebody checked it, so this checks it.
 *
 * ## What "disabled" means here
 *
 * Not turning WSL2 off — that is a system setting, and asking a demo machine to
 * be reconfigured before every rehearsal guarantees the rehearsal stops
 * happening. Instead ARJUN's own detection is told to treat `wsl`, `podman` and
 * `docker` as absent, via `ARJUN_SANDBOX_ASSUME_ABSENT`. Every code path that
 * asks "what isolation does this machine have?" then answers the way it would
 * on a bare laptop, and the whole suite runs down that path.
 *
 * The override can only remove capability, never add it — see
 * `orchestrator::sandbox`. So the rehearsal is strictly more pessimistic than
 * reality, which is the only direction a rehearsal is allowed to be wrong in.
 *
 * ## What a pass means
 *
 * Every test still passes, and the acceptance run reaches the same verdict for
 * criteria 1, 2, 4 and 5. Criterion 3 is *expected* to be BLOCKED here — that is
 * the point of the rehearsal, not a failure of it. A run where criterion 3
 * suddenly passed with every runtime hidden would mean the sandbox check was
 * lying, and this script would say so.
 */

import { spawnSync } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const HIDDEN = 'podman,docker,wsl';

const pathSeparator = () => (process.platform === 'win32' ? ';' : ':');
const cargoBin = () => join(process.env.HOME ?? process.env.USERPROFILE ?? '', '.cargo', 'bin');

function run(command, args) {
  return spawnSync(command, args, {
    cwd: ROOT,
    encoding: 'utf8',
    timeout: 1_800_000,
    // On Windows `npm` is `npm.cmd`, which spawnSync cannot resolve without a
    // shell. Without this the rehearsal reports a frontend build failure that
    // is really a spawn failure — a false alarm, and false alarms are how a
    // rehearsal stops being run.
    shell: process.platform === 'win32',
    env: {
      ...process.env,
      ARJUN_SANDBOX_ASSUME_ABSENT: HIDDEN,
      PATH: `${process.env.PATH}${pathSeparator()}${cargoBin()}`,
    },
  });
}

console.log('\nARJUN rehearsal — as though WSL2 and every container runtime were absent\n');
console.log(`  ARJUN_SANDBOX_ASSUME_ABSENT=${HIDDEN}\n`);

const steps = [];

// 1 — the whole unit suite, down the no-isolation path.
{
  const result = run('cargo', ['test', '--manifest-path', 'src-tauri/Cargo.toml', '--lib']);
  const summary = /test result: (\w+)\. (\d+) passed; (\d+) failed/.exec(result.stdout ?? '');
  steps.push({
    name: 'Unit suite',
    ok: Boolean(summary) && summary[3] === '0',
    detail: summary
      ? `${summary[2]} passed, ${summary[3]} failed`
      : 'the test runner produced no summary — nothing can be concluded',
  });
}

// 2 — the scripted baseline over the five fixtures.
{
  const result = run('cargo', [
    'test',
    '--manifest-path',
    'src-tauri/Cargo.toml',
    '--test',
    'baseline',
  ]);
  const summary = /test result: (\w+)\. (\d+) passed; (\d+) failed/.exec(result.stdout ?? '');
  steps.push({
    name: 'Five-fixture baseline',
    ok: Boolean(summary) && summary[3] === '0',
    detail: summary
      ? `${summary[2]} fixtures passed, ${summary[3]} failed`
      : 'the baseline produced no summary',
  });
}

// 3 — the acceptance criteria, and the shape of the verdict.
{
  const acceptance = run('node', ['scripts/acceptance.mjs']);
  const recordPath = join(ROOT, 'acceptance-baseline.json');

  if (acceptance.error) {
    steps.push({
      name: 'Acceptance criteria',
      ok: false,
      detail: `the acceptance run could not start: ${acceptance.error.message}`,
    });
  } else if (!existsSync(recordPath)) {
    steps.push({
      name: 'Acceptance criteria',
      ok: false,
      detail: 'the acceptance run wrote no baseline record',
    });
  } else {
    const record = JSON.parse(readFileSync(recordPath, 'utf8'));
    const byName = Object.fromEntries(record.results.map((r) => [r.name, r.status]));
    const sandbox = record.results.find((r) => r.name.includes('sandbox'));

    // Criterion 3 must be BLOCKED with every runtime hidden. Anything else
    // means the sandbox check is not actually looking.
    const sandboxHonest = sandbox?.status === 'BLOCKED';
    const othersHold = record.results
      .filter((r) => r !== sandbox)
      .every((r) => r.status === 'PASS' || r.status === 'BLOCKED');

    steps.push({
      name: 'Acceptance criteria',
      ok: sandboxHonest && othersHold,
      detail: sandboxHonest
        ? `criterion 3 correctly reports BLOCKED with no runtime; the rest: ${Object.entries(byName)
            .filter(([n]) => !n.includes('sandbox'))
            .map(([, s]) => s)
            .join(', ')}`
        : `criterion 3 reported ${sandbox?.status ?? 'nothing'} with every runtime hidden — the sandbox check is not looking at the machine`,
    });
  }
}

// 4 — the frontend, which must not have grown a dependency on any of this.
{
  const result = run('npm', ['run', 'build']);
  steps.push({
    name: 'Frontend build',
    ok: result.status === 0,
    detail: result.status === 0 ? 'builds clean' : 'the frontend build failed',
  });
}

console.log('─'.repeat(72));
for (const step of steps) {
  console.log(`  ${step.ok ? '  ok  ' : ' FAIL '}  ${step.name}\n            ${step.detail}`);
}
console.log('─'.repeat(72));

const failed = steps.filter((s) => !s.ok);
if (failed.length === 0) {
  console.log(
    '\n  Nothing on the critical path depends on WSL2 or a container runtime.\n' +
      '  Code execution refuses rather than pretending, and everything else runs.\n',
  );
  process.exit(0);
}

console.log(`\n  ${failed.length} step(s) failed with WSL2 and every container runtime hidden.`);
console.log('  Something on the critical path depends on isolation that may not be there.\n');
process.exit(1);
