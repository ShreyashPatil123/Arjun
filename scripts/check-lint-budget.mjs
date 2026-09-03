#!/usr/bin/env node
/**
 * A ratchet on unused Rust code.
 *
 * ## Why a budget and not `deny(warnings)`
 *
 * `cargo check --all-targets` reports 47 unused imports, fields, constants and
 * functions. Turning them into errors today would mean either a large
 * unrelated cleanup in the same change as everything else, or `#[allow]`
 * scattered over the codebase — and an `#[allow]` added to make a build pass is
 * a warning that has been hidden rather than answered.
 *
 * So the count is pinned instead. It may fall and it may not rise: a change
 * that leaves new dead code behind fails here, and a change that removes some
 * is asked to lower the ceiling. The number goes to zero one honest step at a
 * time, and in the meantime nothing new accumulates.
 *
 * This is the "staged lint gate" — staged in the sense that the standard
 * tightens as the debt is paid, rather than being declared met on day one.
 */

import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

const BUDGET = 'scripts/lint-budget.json';

/** Warnings about code nothing uses, as opposed to style or deprecation. */
const UNUSED =
  /^warning: (unused|(field|constant|function|method|variant|struct|enum|type alias|associated function) .* is never)/;

function unusedWarnings() {
  // `spawnSync` rather than `execFileSync`, because cargo writes every
  // diagnostic to *stderr* and exits zero when they are only warnings. The
  // exec form returns stdout alone, so the first version of this counted zero
  // warnings on a codebase with forty-seven — a gate that passes because it is
  // reading the wrong stream is worse than no gate at all.
  // No `shell: true`. Passing arguments through a shell concatenates rather
  // than escapes them, and there is nothing here that needs one.
  const result = spawnSync(
    process.platform === 'win32' ? 'cargo.exe' : 'cargo',
    ['check', '--manifest-path', 'src-tauri/Cargo.toml', '--all-targets'],
    { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
  );

  if (result.error) {
    console.error(
      `cargo could not be run, so the lint budget could not be measured: ${result.error.message}`,
    );
    process.exit(1);
  }
  if (result.status !== 0) {
    // A compile failure is not this gate's business to report, but it is its
    // business not to pass silently.
    console.error('cargo check failed, so the lint budget could not be measured:\n');
    console.error(result.stderr ?? '');
    process.exit(1);
  }

  const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  return output.split('\n').filter((line) => UNUSED.test(line.trim())).length;
}

function main() {
  const budget = JSON.parse(readFileSync(BUDGET, 'utf8'));
  const ceiling = budget.unusedWarnings;
  const actual = unusedWarnings();

  if (actual > ceiling) {
    console.error(
      `Unused-code budget exceeded: ${actual} warnings, ceiling is ${ceiling}.\n\n` +
        'This change left new dead code behind. Remove it, or — if it is genuinely\n' +
        'needed and unused for a stated reason — say so at the definition rather than\n' +
        `raising the number in ${BUDGET}.`,
    );
    process.exit(1);
  }

  if (actual < ceiling) {
    // Lowered automatically. A ratchet somebody has to remember to tighten is a
    // ratchet that stays where it was.
    budget.unusedWarnings = actual;
    writeFileSync(BUDGET, `${JSON.stringify(budget, null, 2)}\n`);
    console.log(
      `Unused-code budget lowered: ${ceiling} -> ${actual}. Commit the change to ${BUDGET}.`,
    );
    return;
  }

  console.log(`Unused-code budget OK: ${actual} warnings (ceiling ${ceiling}).`);
}

main();
