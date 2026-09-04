#!/usr/bin/env node
/**
 * The offline deployment pack — the installer plus everything it spawns.
 *
 *   npm run pack:offline -- --staging <dir> --out <dir> [--sign-key <pem>]
 *
 * ARJUN's installer contains the code ARJUN wrote. It does not contain Node,
 * CPython or `llama-server`, and it should not: those are third-party
 * redistributables with their own licences, their own signatures and their own
 * CVE clocks, and burying them inside a Tauri bundle would mean this project
 * silently taking on all three. The honest packaging boundary is a *pack* — the
 * signed installer beside the pinned runtimes it needs — and this builds one.
 *
 * ## It fetches nothing
 *
 * There is no download step here, deliberately. `CLAUDE.md` is explicit that
 * nothing on the machine reaches the network without a reviewed decision, and a
 * packaging script that quietly pulls a Node tarball is exactly that decision
 * being made by a script instead of a person. So the operator stages the
 * runtimes themselves, at versions they chose and verified, and this assembles
 * and signs what is on disk.
 *
 * ## It never emits a pack that overstates itself
 *
 * Every component the dependency table declares `External` is required. A
 * staging directory missing one produces an error and no output, not a smaller
 * pack with a cheerful summary — because a pack that is missing `llama-server`
 * and does not say so is worse than no pack at all. The same rule as
 * `scripts/bench.py`: measure, or fail loudly.
 *
 * ## What "signed" means
 *
 * Two layers, and the first matters more:
 *
 * 1. `manifest.json` records the SHA-256 of every file, and `manifest.sha256`
 *    records the digest of the manifest itself. Anyone with `sha256sum` can
 *    verify the whole pack without ARJUN, without this script, and without
 *    trusting either. That is the property `src-tauri/src/package/mod.rs`
 *    argues for, applied to the deployment artifact instead of the task
 *    artifact.
 * 2. `--sign-key` additionally writes a detached Ed25519 signature over the
 *    manifest bytes. Supply the private key as a PKCS#8 PEM. Without the flag
 *    the pack is hashed but unsigned, and `manifest.json` says
 *    `"signed": false` rather than leaving a reader to guess.
 *
 * Asking for a signature and not getting one is an error. There is no path
 * here that writes `"signed": true` without a signature behind it.
 */

import { createHash, sign as signBytes } from 'node:crypto';
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { basename, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const TABLE = join(ROOT, 'src-tauri', 'src', 'deployment', 'mod.rs');

/** The manifest schema version, so a verifier can refuse a shape it predates. */
const FORMAT_VERSION = 1;

// ---------------------------------------------------------------------------
// Arguments

function die(message, extra = []) {
  console.error(`\nOffline pack build FAILED: ${message}\n`);
  for (const line of extra) console.error(`  ${line}`);
  if (extra.length) console.error('');
  process.exit(1);
}

function parseArgs(argv) {
  const args = { staging: null, out: null, signKey: null, force: false };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === '--staging') args.staging = argv[++i];
    else if (flag === '--out') args.out = argv[++i];
    else if (flag === '--sign-key') args.signKey = argv[++i];
    else if (flag === '--force') args.force = true;
    else die(`unrecognised argument ${flag}`);
  }
  return args;
}

// ---------------------------------------------------------------------------
// The component list, derived from the same table the app resolves against

/**
 * The external dependencies the pack must carry, read from the Rust table.
 *
 * Derived rather than duplicated: a dependency added to
 * `src-tauri/src/deployment/mod.rs` becomes a required pack component with no
 * second edit, and so cannot be forgotten here.
 */
function requiredComponents() {
  if (!existsSync(TABLE)) die(`${relative(ROOT, TABLE)} is missing`);
  const source = readFileSync(TABLE, 'utf8');
  const start = source.indexOf('pub const DEPENDENCIES');
  if (start === -1) die('could not find DEPENDENCIES in the deployment table');
  const table = source.slice(start, source.indexOf('\n];', start));

  const components = [];
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
    if ((field('packaging') ?? '') !== 'Packaging::External') continue;
    const id = unquote(field('id'));
    if (!id) continue;
    components.push({
      id,
      program: unquote(field('program')),
      envVar: unquote(field('env_override')),
    });
  }

  if (components.length === 0) {
    die('parsed zero external dependencies; the deployment table shape changed');
  }
  return components;
}

// ---------------------------------------------------------------------------
// Hashing

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

/** Every file under a directory, as forward-slash relative paths. */
function walk(dir, base = dir) {
  const found = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) found.push(...walk(full, base));
    else found.push(relative(base, full).split(sep).join('/'));
  }
  return found;
}

/**
 * Hashes a directory into manifest entries.
 *
 * Sorted, so the manifest is byte-identical for identical input and two builds
 * of the same pack can be compared with `diff`.
 */
function hashTree(dir) {
  return walk(dir)
    .sort()
    .map((path) => {
      const bytes = readFileSync(join(dir, path));
      return { path, sha256: sha256(bytes), bytes: bytes.length };
    });
}

/**
 * Where `tauri build` leaves its output.
 *
 * Checked rather than assumed: an absent bundle means the operator has not run
 * `tauri build`, and saying so is more useful than packing an empty directory.
 */
function findInstaller() {
  const bundleRoot = join(ROOT, 'src-tauri', 'target', 'release', 'bundle');
  if (!existsSync(bundleRoot)) return null;
  const artifacts = walk(bundleRoot).filter((path) =>
    /\.(msi|exe|dmg|deb|AppImage|rpm)$/i.test(path),
  );
  return artifacts.length > 0 ? { root: bundleRoot, artifacts: artifacts.sort() } : null;
}

// ---------------------------------------------------------------------------

const args = parseArgs(process.argv.slice(2));

if (!args.staging || !args.out) {
  die('both --staging and --out are required', [
    'Usage:',
    '  npm run pack:offline -- --staging <dir> --out <dir> [--sign-key <pem>]',
    '',
    'The staging directory holds one subdirectory per external runtime, named',
    'for its dependency id in src-tauri/src/deployment/mod.rs. For this build:',
    ...requiredComponents().map((c) => `  <staging>/${c.id}/    (must contain ${c.program})`),
    '',
    'Stage those yourself, at versions you have chosen and verified. This',
    'script downloads nothing.',
  ]);
}

// `resolve`, not `join`: an operator passing an absolute --staging on Windows
// would otherwise get ROOT glued in front of a drive letter.
const staging = resolve(ROOT, args.staging);
const out = resolve(ROOT, args.out);

if (!existsSync(staging)) die(`the staging directory ${relative(ROOT, staging)} does not exist`);

if (existsSync(out)) {
  if (!args.force) die(`${relative(ROOT, out)} already exists; pass --force to replace it`);
  rmSync(out, { recursive: true, force: true });
}

// The signing key is checked first, so a bad key fails before any work is done
// rather than after a pack has been assembled around it.
let privateKey = null;
if (args.signKey) {
  const keyPath = resolve(ROOT, args.signKey);
  if (!existsSync(keyPath)) die(`the signing key ${relative(ROOT, keyPath)} does not exist`);
  try {
    privateKey = readFileSync(keyPath, 'utf8');
    signBytes(null, Buffer.from('probe'), privateKey);
  } catch (error) {
    die(`the signing key could not be used: ${error.message}`, [
      'The key must be an Ed25519 private key in PKCS#8 PEM form. Generate a pair with:',
      '',
      "  node -e \"const c=require('crypto'),f=require('fs');" +
        "const k=c.generateKeyPairSync('ed25519');" +
        "f.writeFileSync('arjun-pack.key',k.privateKey.export({type:'pkcs8',format:'pem'}));" +
        "f.writeFileSync('arjun-pack.pub',k.publicKey.export({type:'spki',format:'pem'}))\"",
      '',
      'Keep the .key off this machine once the release is cut; ship the .pub.',
    ]);
  }
}

// --- Components -------------------------------------------------------------

const components = requiredComponents();
const missing = [];

for (const component of components) {
  const dir = join(staging, component.id);
  if (!existsSync(dir) || !statSync(dir).isDirectory()) {
    missing.push(`${component.id}: no directory at <staging>/${component.id}`);
    continue;
  }
  const files = walk(dir);
  if (files.length === 0) {
    missing.push(`${component.id}: <staging>/${component.id} is empty`);
    continue;
  }
  // The declared program has to actually be in there, or the pack ships a
  // directory that cannot satisfy the dependency it claims to satisfy.
  const found = files.some(
    (file) => basename(file) === component.program || basename(file) === `${component.program}.exe`,
  );
  if (!found) {
    missing.push(
      `${component.id}: no "${component.program}" (or ${component.program}.exe) under ` +
        `<staging>/${component.id}`,
    );
  }
}

if (missing.length > 0) {
  die(`the staging directory is incomplete (${missing.length} problem(s))`, [
    ...missing.map((line) => `- ${line}`),
    '',
    'A pack missing a runtime would install and then fail at first use, which is',
    'the failure this pack exists to prevent. Nothing was written.',
  ]);
}

const installer = findInstaller();
if (!installer) {
  die('no built installer found under src-tauri/target/release/bundle', [
    'Run a release build first, for example:',
    '  npm run tauri:build:gpu      (CUDA)',
    '  npm run tauri:build:vulkan   (Vulkan)',
    '',
    'A pack without the installer is not a deployment pack.',
  ]);
}

// --- Assemble ---------------------------------------------------------------

mkdirSync(out, { recursive: true });

const manifestComponents = [];

const installerOut = join(out, 'installer');
mkdirSync(installerOut, { recursive: true });
for (const artifact of installer.artifacts) {
  cpSync(join(installer.root, artifact), join(installerOut, basename(artifact)));
}
manifestComponents.push({
  id: 'installer',
  role: 'The ARJUN application, containing the agent runtime, skills, agents and sidecars.',
  source: relative(ROOT, installer.root).split(sep).join('/'),
  files: hashTree(installerOut),
});

for (const component of components) {
  const from = join(staging, component.id);
  const to = join(out, component.id);
  cpSync(from, to, { recursive: true });
  manifestComponents.push({
    id: component.id,
    role: `Supplies ${component.program}; the app finds it through ${component.envVar}.`,
    source: relative(ROOT, from).split(sep).join('/'),
    program: component.program,
    envVar: component.envVar,
    files: hashTree(to),
  });
}

// --- The environment script -------------------------------------------------
//
// The pack is only useful if the installed app actually finds it. Rather than
// asking an operator to set three variables by hand and get one of them wrong,
// the pack carries the script that sets them, generated from the same table.

const envLines = components.map((component) => {
  const dir = join(out, component.id);
  const program = walk(dir).find(
    (file) => basename(file) === component.program || basename(file) === `${component.program}.exe`,
  );
  return { component, program };
});

writeFileSync(
  join(out, 'arjun-env.cmd'),
  [
    '@echo off',
    'REM Points ARJUN at the runtimes in this pack.',
    'REM Generated by scripts/build-offline-pack.mjs -- do not edit by hand.',
    'REM Run this in the shell that launches ARJUN, or set these machine-wide.',
    'set "ARJUN_PACK=%~dp0"',
    ...envLines.map(
      ({ component, program }) =>
        `set "${component.envVar}=%ARJUN_PACK%${component.id}\\${program.split('/').join('\\')}"`,
    ),
    '',
  ].join('\r\n'),
);

writeFileSync(
  join(out, 'arjun-env.sh'),
  [
    '#!/bin/sh',
    '# Points ARJUN at the runtimes in this pack.',
    '# Generated by scripts/build-offline-pack.mjs -- do not edit by hand.',
    'ARJUN_PACK="$(cd "$(dirname "$0")" && pwd)"',
    ...envLines.map(
      ({ component, program }) =>
        `export ${component.envVar}="$ARJUN_PACK/${component.id}/${program}"`,
    ),
    '',
  ].join('\n'),
);

// --- Manifest ---------------------------------------------------------------

const manifest = {
  formatVersion: FORMAT_VERSION,
  product: 'ARJUN',
  version: JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).version,
  createdAt: new Date().toISOString(),
  // Stated plainly so a reader never has to infer it from the presence of a
  // file. An unsigned pack says so about itself.
  signed: Boolean(privateKey),
  signatureAlgorithm: privateKey ? 'ed25519' : null,
  components: manifestComponents,
  verify: {
    manifestDigest: 'manifest.sha256 holds the SHA-256 of manifest.json',
    withoutArjun:
      'Every entry above carries its own SHA-256. The pack can be verified with ' +
      'sha256sum alone; nothing in the check requires ARJUN to be installed or trusted.',
  },
};

const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
writeFileSync(join(out, 'manifest.json'), manifestBytes);

const manifestDigest = sha256(manifestBytes);
writeFileSync(join(out, 'manifest.sha256'), `${manifestDigest}  manifest.json\n`);

if (privateKey) {
  writeFileSync(
    join(out, 'manifest.sig'),
    `${signBytes(null, manifestBytes, privateKey).toString('base64')}\n`,
  );
}

// --- Report -----------------------------------------------------------------

const fileCount = manifestComponents.reduce((sum, c) => sum + c.files.length, 0);
const byteCount = manifestComponents.reduce(
  (sum, c) => sum + c.files.reduce((inner, f) => inner + f.bytes, 0),
  0,
);

console.log(`\nOffline deployment pack written to ${relative(ROOT, out)}\n`);
for (const component of manifestComponents) {
  console.log(`  ${component.id.padEnd(16)} ${String(component.files.length).padStart(5)} file(s)`);
}
console.log(
  `\n  ${fileCount} files, ${(byteCount / 1_048_576).toFixed(1)} MiB` +
    `\n  manifest.json  sha256 ${manifestDigest}`,
);
console.log(
  privateKey
    ? '  manifest.sig   ed25519 detached signature over manifest.json'
    : '  NOT SIGNED     pass --sign-key to write a detached signature',
);
console.log(
  '\nRehearse it before shipping it:\n' +
    `  npm run rehearse:offline-install -- --pack ${relative(ROOT, out)}\n`,
);
