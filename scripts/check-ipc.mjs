#!/usr/bin/env node
/**
 * The IPC surface, against what claims to own it.
 *
 * ## Why this exists
 *
 * A Tauri command is a function any page can call. There were 123 of them and
 * no record of who called which — so sixteen were registered with no consumer
 * at all, and nothing would have noticed a seventeenth. In the other direction,
 * a frontend service wrapping ten commands survived their removal because
 * nothing checks that an `invoke` names a handler that exists; every call would
 * have failed at runtime, in front of whoever was using it.
 *
 * This closes both directions against `src-tauri/ipc-manifest.json`:
 *
 *  - every registered command has an entry, with an owner and a stated consumer;
 *  - every entry names a command that is still registered;
 *  - every `invoke("…")` in `src/` names a registered command;
 *  - every command the manifest calls `frontend` really is called from `src/`.
 *
 * The last one matters most. "frontend" is the claim that is easiest to make
 * and easiest to leave behind when the caller is deleted, which is exactly how
 * the sixteen accumulated.
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const MANIFEST = 'src-tauri/ipc-manifest.json';
const LIB = 'src-tauri/src/lib.rs';
const FRONTEND = 'src';

/** The commands `generate_handler!` registers, by name. */
function registeredCommands() {
  const lib = readFileSync(LIB, 'utf8');
  const block = /generate_handler!\[(.*?)\]\)/s.exec(lib);
  if (!block) {
    throw new Error(`Could not find the generate_handler! block in ${LIB}.`);
  }
  return block[1]
    .split('\n')
    .map((line) => line.trim().replace(/,$/, ''))
    .filter((line) => line.length > 0 && !line.startsWith('//'))
    .map((path) => path.split('::').pop());
}

/** Every command name the frontend invokes. */
function invokedCommands() {
  const found = new Set();
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) {
        walk(path);
        continue;
      }
      if (!/\.(ts|tsx)$/.test(entry)) continue;
      const text = readFileSync(path, 'utf8');
      for (const match of text.matchAll(/invoke(?:<[^>]*>)?\(\s*['"]([a-z0-9_]+)['"]/g)) {
        found.add(match[1]);
      }
    }
  };
  walk(FRONTEND);
  return found;
}

const CONSUMERS = new Set(['frontend', 'admin', 'external', 'frontend-pending']);

function main() {
  const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
  const declared = manifest.commands ?? {};
  const registered = registeredCommands();
  const invoked = invokedCommands();
  const problems = [];

  for (const command of registered) {
    const entry = declared[command];
    if (!entry) {
      problems.push(
        `${command} is registered in lib.rs and has no entry in ${MANIFEST}. ` +
          'Add one naming its owner and which of frontend/admin/external/frontend-pending ' +
          'consumes it, and why.',
      );
      continue;
    }
    if (!entry.owner || entry.owner === 'unassigned') {
      problems.push(`${command} has no owner in ${MANIFEST}.`);
    }
    if (!CONSUMERS.has(entry.consumer)) {
      problems.push(
        `${command} declares consumer "${entry.consumer}", which is not one of ` +
          `${[...CONSUMERS].join(', ')}.`,
      );
    }
    if (!entry.note || entry.note.trim().length === 0) {
      problems.push(`${command} has no note saying why it exists.`);
    }
    // The claim that rots. A command whose only caller was deleted keeps
    // saying "frontend" until something asks.
    if (entry.consumer === 'frontend' && !invoked.has(command)) {
      problems.push(
        `${command} is declared "frontend" and nothing in ${FRONTEND}/ invokes it. ` +
          'Either wire it up, or change its consumer to admin/external/frontend-pending ' +
          'and say why it is registered.',
      );
    }
  }

  const registeredSet = new Set(registered);
  for (const command of Object.keys(declared)) {
    if (!registeredSet.has(command)) {
      problems.push(
        `${MANIFEST} lists ${command}, which lib.rs no longer registers. Remove the entry.`,
      );
    }
  }

  for (const command of invoked) {
    if (!registeredSet.has(command)) {
      problems.push(
        `The frontend invokes ${command} and lib.rs does not register it. ` +
          'Every call would fail at runtime.',
      );
    }
  }

  if (problems.length > 0) {
    console.error('IPC contract check failed:\n');
    for (const problem of problems) console.error(`  - ${problem}`);
    console.error(`\n${problems.length} problem(s).`);
    process.exit(1);
  }

  const byConsumer = {};
  for (const entry of Object.values(declared)) {
    byConsumer[entry.consumer] = (byConsumer[entry.consumer] ?? 0) + 1;
  }
  const summary = Object.entries(byConsumer)
    .sort()
    .map(([consumer, count]) => `${count} ${consumer}`)
    .join(', ');
  console.log(`IPC contract OK: ${registered.length} commands (${summary}).`);
}

main();
