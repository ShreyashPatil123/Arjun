#!/usr/bin/env node
/**
 * Builds and verifies the agent-system fixture manifest.
 *
 *   node scripts/fixture-manifest.mjs            # rebuild the manifest
 *   node scripts/fixture-manifest.mjs --check    # fail if it is stale
 *
 * ## Why the manifest is generated rather than written
 *
 * A fixture pack's whole value is that the inputs did not change between the
 * run that produced a score and the run that reproduces it. Hand-maintained
 * hashes stop being true the first time somebody edits a fixture and forgets,
 * and a stale hash is worse than none: it asserts an integrity that is not
 * there. So the hashes come from the files, and `--check` is what makes the
 * assertion survive.
 *
 * ## Reuse, recorded rather than copied
 *
 * `src-tauri/tests/fixtures/inspection-report.md` and `maintenance-sop.md`
 * already exist and are sound. They are listed here at their existing paths
 * with their existing hashes, not copied into the pack. Two copies of one SOP
 * is two SOPs, and the next person to edit one of them will edit the wrong one.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PACK = join(ROOT, 'fixtures', 'agent-system', 'v1');
const MANIFEST = join(PACK, 'manifest.json');
const CHECK = process.argv.includes('--check');

const VERSION = '1.0.0';

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const rel = (path) => relative(ROOT, path).split(sep).join(posix.sep);

/**
 * Directories and files that are build output, not fixtures.
 *
 * `__pycache__` is the one that actually happened: running the hidden tests
 * once put a `.pyc` in the pack, the manifest hashed it, and the next Python
 * version would have made `--check` fail for a file no case ever reads.
 */
const NOT_A_FIXTURE = new Set(['__pycache__', '.pytest_cache', '.DS_Store', 'Thumbs.db']);

/** Every file under a directory, sorted, so the manifest is order-stable. */
function walk(dir, out = []) {
  for (const name of readdirSync(dir).sort()) {
    if (NOT_A_FIXTURE.has(name)) continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

function entry(path, role, note) {
  return {
    path: rel(path),
    role,
    bytes: statSync(path).size,
    sha256: sha256(path),
    ...(note ? { note } : {}),
  };
}

// ── Reused fixtures, in place ───────────────────────────────────────────────

const REUSED = [
  [join(ROOT, 'src-tauri', 'tests', 'fixtures', 'inspection-report.md'),
    'source.inspection-report',
    'Pre-existing and sound. The Markdown ground truth behind the scanned pages: an extraction graded against ' +
    'the scans is graded against this text.'],
  [join(ROOT, 'src-tauri', 'tests', 'fixtures', 'maintenance-sop.md'),
    'source.sop.revision-c',
    'Pre-existing and sound. Revision C, one half of the conflicting pair. Left where it is so the two ' +
    'revisions cannot drift apart in two directories.'],
  [join(ROOT, 'src-tauri', 'tests', 'fixtures', 'pid-excerpt.png'),
    'source.drawing',
    'Pre-existing. A P&ID excerpt for the engineering-drawing input class named in PS-E. It carries no known ' +
    'text ground truth, so it grades presence-of-handling, not accuracy.'],
];

// ── Assembly ────────────────────────────────────────────────────────────────

const roleOf = (path) => {
  const r = rel(path);
  if (r.includes('/expected/')) return 'expected.held-out';
  if (r.includes('/failures/')) return 'failure-case';
  if (r.includes('/sources/scan/')) return r.endsWith('.py') ? 'generator' : 'source.scan';
  if (r.includes('/sources/sop/')) return 'source.sop.revision-d';
  if (r.includes('/sources/calc/')) return 'source.calculation';
  if (r.includes('/sources/briefs/')) return 'source.brief';
  if (r.includes('/sources/code/')) return 'source.code-task';
  return 'pack';
};

const files = walk(PACK)
  .filter((path) => path !== MANIFEST)
  .map((path) => entry(path, roleOf(path)));

const reused = REUSED
  .filter(([path]) => existsSync(path))
  .map(([path, role, note]) => entry(path, role, note));

const missingReuse = REUSED.filter(([path]) => !existsSync(path)).map(([path]) => rel(path));

const manifest = {
  schema: 'arjun.fixture-manifest/1',
  pack: 'agent-system',
  version: VERSION,
  // No generated_at. A timestamp would make every rebuild a diff, and --check
  // could never tell a stale manifest from a fresh one.
  confidentiality:
    'Nonconfidential throughout (PS-K). Every vessel tag, reading, name and certificate number is invented. ' +
    'Nothing here came from a plant, a vendor or a customer.',
  licence: 'Same licence as the repository. The pack contains no third-party content.',
  provenance: {
    scans:
      'Rendered by fixtures/agent-system/v1/sources/scan/make-scans.py from the text of the reused inspection ' +
      'report. Deterministic: seed 26117, no timestamps, byte-identical on re-run.',
    sop_revision_d:
      'Written for this pack. It contradicts Revision C on purpose — see its section 3.1.',
    code_task:
      'Written for this pack, with its grading tests held out under expected/hidden-tests/.',
  },
  held_out: {
    directory: 'fixtures/agent-system/v1/expected',
    why: 'Grading keys an evaluated agent must not read. See that directory README.',
    enforced_by:
      'scripts/agent-baseline.mjs asserts no expected/ path resolves inside a run workspace root before it ' +
      'hands any workspace out, and reports BLOCKED rather than grading against a readable key.',
  },
  regenerate: {
    scans: 'python fixtures/agent-system/v1/sources/scan/make-scans.py',
    manifest: 'node scripts/fixture-manifest.mjs',
    check: 'node scripts/fixture-manifest.mjs --check',
  },
  reused_in_place: reused,
  files,
};

if (missingReuse.length) {
  manifest.reuse_unresolved = missingReuse.map((path) => ({
    path,
    consequence:
      'This fixture is referenced by the pack and is not on this machine. Cases depending on it are BLOCKED, ' +
      'not failed.',
  }));
}

const serialised = `${JSON.stringify(manifest, null, 2)}\n`;

if (CHECK) {
  if (!existsSync(MANIFEST)) {
    process.stderr.write('The fixture manifest does not exist. Run: node scripts/fixture-manifest.mjs\n');
    process.exit(1);
  }
  if (readFileSync(MANIFEST, 'utf8') !== serialised) {
    process.stderr.write(
      'The fixture manifest is stale: a file in the pack changed and its hash was not updated.\n' +
      'Run `node scripts/fixture-manifest.mjs` and commit the result.\n');
    process.exit(1);
  }
  process.stdout.write(
    `fixture manifest current: ${files.length} pack file(s), ${reused.length} reused in place\n`);
  process.exit(0);
}

writeFileSync(MANIFEST, serialised, 'utf8');
process.stdout.write(
  `fixture manifest written to ${rel(MANIFEST)}\n` +
  `  pack files ${files.length}; reused in place ${reused.length}` +
  (missingReuse.length ? `; unresolved ${missingReuse.length}` : '') + '\n');
