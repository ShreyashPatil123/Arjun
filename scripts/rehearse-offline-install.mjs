#!/usr/bin/env node
/**
 * The clean-machine rehearsal.
 *
 *   npm run rehearse:offline-install
 *   npm run rehearse:offline-install -- --pack dist/pack [--pubkey arjun-pack.pub]
 *
 * `scripts/rehearse-no-wsl.mjs` next door rehearses a machine with no container
 * runtime. This rehearses the other demo-day disaster: a machine with no
 * developer toolchain and no network, where the only things present are the
 * installer and whatever the deployment pack brought with it.
 *
 * ## Why the staged resource tree is the thing to inspect
 *
 * `tauri build` copies everything in `bundle.resources` into
 * `src-tauri/target/release/_up_/` — `_up_` being how it spells the leading
 * `../` — and it is *that* tree the MSI and the NSIS installer are built from.
 * So asking "is the document sidecar in `_up_/sidecars/`?" is the same question
 * as "will a clean installation have it?", answerable without extracting a
 * 278 MB MSI or installing anything.
 *
 * This is the check that would have caught the original bug. Before the fix,
 * `_up_/` held `agent-runtime`, `agents` and `skills`, and nothing else — the
 * three Python sidecars had never been in an installer at all.
 *
 * ## What a pass means, and what it does not
 *
 * A pass means: the installer payload physically contains every file the code
 * resolves as bundled, the pack's hashes match its manifest, and every external
 * runtime is reachable from the pack rather than from this machine's PATH.
 *
 * It does **not** mean the installer has been run on a clean Windows box. This
 * script cannot create a VM, and pretending otherwise would be the exact
 * failure this repository's rules are written against. The closing report says
 * plainly what remains unproven, and that step is a human one.
 */

import { createHash, verify as verifyBytes } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { delimiter, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const TABLE = join(ROOT, 'src-tauri', 'src', 'deployment', 'mod.rs');
const STAGED = join(ROOT, 'src-tauri', 'target', 'release', '_up_');

const failures = [];
const warnings = [];
const passes = [];

const fail = (m) => failures.push(m);
const warn = (m) => warnings.push(m);
const pass = (m) => passes.push(m);

// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const args = { pack: null, pubkey: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--pack') args.pack = argv[++i];
    else if (argv[i] === '--pubkey') args.pubkey = argv[++i];
    else {
      console.error(`unrecognised argument ${argv[i]}`);
      process.exit(2);
    }
  }
  return args;
}

/** The dependency table, read from the Rust source the app resolves against. */
function readDependencies() {
  const source = readFileSync(TABLE, 'utf8');
  const start = source.indexOf('pub const DEPENDENCIES');
  if (start === -1) {
    fail('could not find DEPENDENCIES in the deployment table');
    return [];
  }
  const table = source.slice(start, source.indexOf('\n];', start));
  const entries = [];
  for (const block of table.split(/Dependency\s*\{/).slice(1)) {
    const field = (name) => {
      const match = block.match(new RegExp(`${name}:\\s*([^\\n]+?),\\s*\\n`));
      return match ? match[1].trim() : null;
    };
    const unquote = (raw) => {
      if (!raw || raw === 'None') return null;
      const inner = raw.startsWith('Some(') ? raw.slice(5, -1) : raw;
      const match = inner.match(/^"(.*)"$/s);
      return match ? match[1] : null;
    };
    const id = unquote(field('id'));
    if (!id) continue;
    entries.push({
      id,
      label: unquote(field('label')) ?? id,
      packaging: (field('packaging') ?? '').replace('Packaging::', ''),
      bundlePath: unquote(field('bundle_path')),
      program: unquote(field('program')),
      envVar: unquote(field('env_override')),
    });
  }
  if (entries.length === 0) fail('parsed zero dependencies; the table shape changed');
  return entries;
}

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function walk(dir, base = dir) {
  const found = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) found.push(...walk(full, base));
    else found.push(relative(base, full).split(sep).join('/'));
  }
  return found;
}

// ---------------------------------------------------------------------------

const args = parseArgs(process.argv.slice(2));
const dependencies = readDependencies();

console.log('\nClean-machine rehearsal\n=======================\n');

// --- Step 1: the installer payload -----------------------------------------
//
// The decisive check. Everything else is supporting evidence.

console.log('1. Installer payload');

if (!existsSync(STAGED)) {
  fail(
    `no staged resource tree at ${relative(ROOT, STAGED)}. Run a release build first ` +
      '(npm run tauri:build:gpu or tauri:build:vulkan). Without it this rehearsal ' +
      'cannot see what an installation would contain, and a rehearsal that guesses ' +
      'is worse than none.',
  );
} else {
  const staged = walk(STAGED);
  const stagedSet = new Set(staged);

  for (const dep of dependencies) {
    if (dep.packaging !== 'Bundled') continue;
    if (stagedSet.has(dep.bundlePath)) {
      pass(`${dep.label} ships: _up_/${dep.bundlePath}`);
    } else {
      fail(
        `${dep.label} is NOT in the installer payload. The code resolves ` +
          `"${dep.bundlePath}"; the staged tree does not contain it. A clean ` +
          `installation would fail at "${dep.label}" and fall back to a checkout ` +
          'that is not on the target machine. Add it to bundle.resources in ' +
          'tauri.conf.json and rebuild.',
      );
    }
  }

  // The staged tree is only as current as the last build.
  const configured =
    JSON.parse(readFileSync(join(ROOT, 'src-tauri', 'tauri.conf.json'), 'utf8')).bundle
      ?.resources ?? [];
  const stale = configured
    .map((entry) => entry.replace(/^\.\.\//, '').replace(/\/$/, ''))
    .filter((entry) => !staged.some((file) => file.startsWith(entry)));
  if (stale.length > 0) {
    warn(
      'tauri.conf.json declares resources the staged tree does not contain ' +
        `(${stale.join(', ')}). The build predates the configuration — rebuild before ` +
        'trusting this rehearsal.',
    );
  }
}

// --- Step 2: the deployment pack -------------------------------------------

console.log('2. Deployment pack');

let packDir = null;
if (!args.pack) {
  warn(
    'no --pack given, so the external runtimes (node, python, llama-server) were not ' +
      'verified. Build one with `npm run pack:offline` and pass it here.',
  );
} else {
  packDir = resolve(ROOT, args.pack);
  const manifestPath = join(packDir, 'manifest.json');
  if (!existsSync(manifestPath)) {
    fail(`${relative(ROOT, manifestPath)} does not exist; that is not a deployment pack`);
    packDir = null;
  } else {
    const manifestBytes = readFileSync(manifestPath);
    const manifest = JSON.parse(manifestBytes.toString('utf8'));

    const digestPath = join(packDir, 'manifest.sha256');
    if (!existsSync(digestPath)) {
      fail('manifest.sha256 is missing; the pack cannot be checked against itself');
    } else {
      const recorded = readFileSync(digestPath, 'utf8').trim().split(/\s+/)[0];
      const actual = sha256(manifestBytes);
      if (recorded === actual) {
        pass(`manifest.json matches manifest.sha256 (${actual.slice(0, 16)}...)`);
      } else {
        fail(`manifest.json has been modified: recorded ${recorded}, actual ${actual}`);
      }
    }

    if (manifest.signed) {
      const sigPath = join(packDir, 'manifest.sig');
      if (!existsSync(sigPath)) {
        fail('the manifest says "signed": true but manifest.sig is missing');
      } else if (!args.pubkey) {
        warn(
          'the pack is signed but no --pubkey was given, so the signature was not ' +
            'checked. A signature nobody verifies is decoration.',
        );
      } else {
        const keyPath = resolve(ROOT, args.pubkey);
        if (!existsSync(keyPath)) {
          fail(`the public key ${relative(ROOT, keyPath)} does not exist`);
        } else {
          const signature = Buffer.from(readFileSync(sigPath, 'utf8').trim(), 'base64');
          const ok = verifyBytes(null, manifestBytes, readFileSync(keyPath, 'utf8'), signature);
          if (ok) pass('manifest.sig verifies against the supplied public key');
          else fail('manifest.sig does NOT verify against the supplied public key');
        }
      }
    } else {
      warn('the pack is unsigned (manifest says "signed": false)');
    }

    let checked = 0;
    let corrupt = 0;
    for (const component of manifest.components ?? []) {
      for (const file of component.files ?? []) {
        const full = join(packDir, component.id, ...file.path.split('/'));
        if (!existsSync(full)) {
          fail(`${component.id}/${file.path} is in the manifest but missing from the pack`);
          corrupt += 1;
          continue;
        }
        checked += 1;
        if (sha256(readFileSync(full)) !== file.sha256) {
          fail(`${component.id}/${file.path} does not match its recorded SHA-256`);
          corrupt += 1;
        }
      }
    }
    if (corrupt === 0 && checked > 0) pass(`all ${checked} pack files match their recorded hashes`);
  }
}

// --- Step 3: resolution on a machine with nothing installed -----------------

console.log('3. Resolution with a stripped PATH');

/**
 * A PATH with nothing but the operating system on it.
 *
 * The point is to answer "does this dependency come from the pack, or from this
 * developer's machine?" — a question that cannot be asked while the real PATH
 * still has Node and Python on it.
 */
function systemOnlyPath() {
  if (process.platform === 'win32') {
    const root = process.env.SystemRoot ?? 'C:\\Windows';
    return [root, join(root, 'System32')].join(delimiter);
  }
  return ['/usr/bin', '/bin'].join(delimiter);
}

function findOnPath(program, searchPath) {
  const extensions = process.platform === 'win32' ? ['.exe', '.cmd', '.bat', ''] : [''];
  for (const dir of searchPath.split(delimiter).filter(Boolean)) {
    for (const extension of extensions) {
      const candidate = join(dir, `${program}${extension}`);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

const stripped = systemOnlyPath();

for (const dep of dependencies) {
  if (dep.packaging !== 'External') continue;

  let fromPack = null;
  if (packDir) {
    const componentDir = join(packDir, dep.id);
    if (existsSync(componentDir)) {
      const match = walk(componentDir).find((file) => {
        const name = file.split('/').pop();
        return name === dep.program || name === `${dep.program}.exe`;
      });
      if (match) fromPack = join(componentDir, ...match.split('/'));
    }
  }

  const fromSystem = findOnPath(dep.program, stripped);

  if (fromPack) {
    pass(`${dep.label} comes from the pack (${dep.envVar}=${relative(ROOT, fromPack)})`);
  } else if (fromSystem) {
    pass(`${dep.label} is part of the operating system (${fromSystem})`);
  } else {
    fail(
      `${dep.label} would not be found on a clean machine. With a system-only PATH ` +
        `there is no "${dep.program}", and the pack does not supply one. Stage it under ` +
        `<staging>/${dep.id}/ and rebuild the pack, or the installed app will fail the ` +
        `first time it needs ${dep.label}.`,
    );
  }
}

// --- Report -----------------------------------------------------------------

console.log('');
for (const line of passes) console.log(`  ok    ${line}`);
for (const line of warnings) console.log(`  warn  ${line}`);

if (failures.length > 0) {
  console.error(`\nRehearsal FAILED with ${failures.length} problem(s):\n`);
  for (const line of failures) console.error(`  x  ${line}\n`);
  process.exit(1);
}

console.log(
  '\nRehearsal passed.\n\n' +
    'What this proved: the installer payload contains every bundled dependency the\n' +
    'code resolves, the pack matches its manifest, and each external runtime comes\n' +
    'from the pack or from the OS rather than from this machine.\n\n' +
    'What it did NOT prove: that the installer runs. This script does not create a\n' +
    'clean Windows VM and does not execute the MSI. Before shipping, install the\n' +
    'pack on a network-disabled Windows machine that has never had a toolchain, run\n' +
    'arjun-env.cmd, launch ARJUN, and open a PDF — that last step is the one no\n' +
    'script here can do for you.\n',
);
