#!/usr/bin/env node
/**
 * Offline build gate — proves ARJUN can be built with no network at all.
 *
 * The egress gate next door proves no *runtime* code can call out. This proves
 * the other half: that the build itself needs nothing fetched. Together they
 * cover the case a demo cannot otherwise rule out — a dependency that quietly
 * reaches the internet the first time someone builds on a fresh machine.
 *
 * It is deliberately not wired into `npm run build`: it is minutes, not seconds,
 * and re-verifying it on every incremental build would only train people to skip
 * it. Run it before packaging, and in CI.
 *
 * A failure here is usually one of two things, and the output says which:
 *   - a crate or package missing from the local cache, meaning the offline
 *     install set is incomplete; or
 *   - a build script that fetches at compile time, which has to be vendored.
 */

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { delimiter, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');

/** Each step must succeed using only what is already on disk. */
const STEPS = [
  {
    name: 'Rust dependencies resolve from the local registry',
    command: 'cargo',
    args: ['check', '--offline', '--lib'],
    cwd: join(ROOT, 'src-tauri'),
    hint: 'Run `cargo fetch` once on a connected machine, then retry offline.',
  },
  {
    name: 'Node dependencies resolve from the npm cache',
    command: 'npm',
    args: ['ci', '--offline', '--dry-run', '--no-audit', '--no-fund'],
    cwd: ROOT,
    hint: 'Run `npm ci` once on a connected machine to populate the cache, then retry.',
  },
  {
    name: 'TypeScript compiles',
    command: 'npx',
    args: ['--offline', 'tsc', '--noEmit'],
    cwd: ROOT,
    hint: 'Type errors are unrelated to the network — fix them first.',
  },
];

if (!existsSync(join(ROOT, 'src-tauri', 'Cargo.toml'))) {
  console.error('offline gate: cannot find src-tauri/Cargo.toml — run this from the repo.');
  process.exit(1);
}

/**
 * A toolchain installed but not on PATH is common on Windows, where an
 * installer adds a directory the current shell never picked up. Looking in the
 * usual place first means this gate assesses the build rather than the shell.
 */
function withToolchainsOnPath() {
  const extra = [join(homedir(), '.cargo', 'bin')].filter(existsSync);
  if (extra.length === 0) return process.env;
  return { ...process.env, PATH: [process.env.PATH, ...extra].join(delimiter) };
}

const ENVIRONMENT = withToolchainsOnPath();

/**
 * Whether a command can be run at all.
 *
 * On Windows a missing command surfaces as a non-zero exit from the shell
 * rather than a spawn error, so `result.error` alone does not detect it. Probing
 * with `--version` separates "this tool is absent" from "this tool ran and
 * reported a problem" — a distinction this gate depends on, because reporting a
 * missing toolchain as an offline failure is both false and alarming.
 */
function isRunnable(command) {
  const probe = spawnSync(command, ['--version'], {
    shell: process.platform === 'win32',
    encoding: 'utf8',
    env: ENVIRONMENT,
  });
  return !probe.error && probe.status === 0;
}

let failed = 0;
let unverifiable = 0;

for (const step of STEPS) {
  process.stdout.write(`  ${step.name} … `);

  if (!isRunnable(step.command)) {
    unverifiable += 1;
    console.log('CANNOT VERIFY');
    console.log(`      ${step.command} is not installed or not reachable from this shell.`);
    console.log('      This says nothing about whether the build needs the network.');
    continue;
  }

  const result = spawnSync(step.command, step.args, {
    cwd: step.cwd,
    // `shell` so `npm` and `npx` resolve through their Windows shims.
    shell: process.platform === 'win32',
    encoding: 'utf8',
    env: ENVIRONMENT,
  });

  if (result.status === 0) {
    console.log('ok');
    continue;
  }

  failed += 1;
  console.log('FAILED');
  const detail = `${result.stderr ?? ''}${result.stdout ?? ''}`
    .split(/\r?\n/)
    .filter(Boolean)
    .slice(-8);
  for (const line of detail) console.log(`      ${line}`);
  console.log(`      ${step.hint}`);
}

if (failed === 0 && unverifiable === 0) {
  console.log('\noffline gate: pass — the build needs no network');
  process.exit(0);
}

if (failed > 0) {
  console.error(`\noffline gate: FAIL — ${failed} step(s) genuinely could not complete offline`);
} else {
  console.error(
    `\noffline gate: INCONCLUSIVE — ${unverifiable} step(s) could not be checked because ` +
      'their toolchain is missing. Install it and run this again; do not read this as a pass.',
  );
}
process.exit(1);
