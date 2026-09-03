#!/usr/bin/env node
/**
 * Frontend modules nothing can reach.
 *
 * ## Why this exists
 *
 * Twelve modules under `src/` had no importer. Two of them were worse than
 * merely unused:
 *
 *  - `types/model-providers.ts` declared `ModelProvider {}`, `ModelMetadata {}`
 *    and `ProviderType {}` — three empty interfaces that constrain nothing and
 *    imply a typed provider abstraction that was never built.
 *  - `services/recommendation.service.ts` and `services/packManager.service.ts`
 *    were the only callers of eight Tauri commands. Because they existed, the
 *    IPC manifest recorded those commands as frontend-consumed. Dead code was
 *    vouching for live surface.
 *
 * A module with no importer is not always a defect — a barrel imported by
 * directory path, a lazily imported component, an ambient declaration file and
 * a test all look unreferenced to a naive scan, and this walks the real import
 * graph so they do not. What it catches is the module that genuinely nothing
 * can reach, which is the one that quietly accumulates and then starts making
 * claims on behalf of code that runs.
 */

import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';

const ROOT = 'src';
/** Where execution begins. Everything live is reachable from one of these. */
const ENTRY_POINTS = ['src/main.tsx', 'src/App.tsx'];
const EXTENSIONS = ['', '.ts', '.tsx', '/index.ts', '/index.tsx'];

/** Every source file under `src/`, excluding tests and ambient declarations. */
function allModules() {
  const found = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) {
        walk(path);
        continue;
      }
      if (!/\.(ts|tsx)$/.test(entry)) continue;
      // Tests are reached by the runner, not by an import. Ambient `.d.ts`
      // files are reached by the compiler.
      if (/\.test\.tsx?$/.test(entry) || /\.d\.ts$/.test(entry)) continue;
      if (path.split(/[\\/]/).includes('__tests__')) continue;
      found.push(path.replace(/\\/g, '/'));
    }
  };
  walk(ROOT);
  return found;
}

/** Resolves one import specifier to a file, or null when it leaves `src/`. */
function resolveSpecifier(fromFile, specifier) {
  if (!specifier.startsWith('.')) return null;
  const base = resolve(dirname(fromFile), specifier.replace(/\.js$/, ''));
  for (const extension of EXTENSIONS) {
    const candidate = `${base}${extension}`;
    if (existsSync(candidate) && statSync(candidate).isFile()) {
      return relative(process.cwd(), candidate).replace(/\\/g, '/');
    }
  }
  return null;
}

/** Every relative specifier a file imports, static or lazy. */
function specifiersIn(text) {
  const found = [];
  // `import … from '…'`, bare `import '…'`, and `import('…')` for lazy routes
  // and `React.lazy`.
  for (const pattern of [
    /from\s+['"](\.[^'"]+)['"]/g,
    /import\s+['"](\.[^'"]+)['"]/g,
    /import\(\s*['"](\.[^'"]+)['"]\s*\)/g,
  ]) {
    for (const match of text.matchAll(pattern)) found.push(match[1]);
  }
  return found;
}

function main() {
  const modules = allModules();
  const reachable = new Set();
  const queue = ENTRY_POINTS.filter((entry) => existsSync(entry));

  if (queue.length === 0) {
    console.error(`No entry point found. Looked for: ${ENTRY_POINTS.join(', ')}`);
    process.exit(1);
  }

  while (queue.length > 0) {
    const file = queue.pop();
    if (reachable.has(file)) continue;
    reachable.add(file);
    const text = readFileSync(file, 'utf8');
    for (const specifier of specifiersIn(text)) {
      const target = resolveSpecifier(file, specifier);
      if (target && !reachable.has(target)) queue.push(target);
    }
  }

  // A module a *test* imports is reachable for a reason worth keeping: the
  // thing under test. Those are collected in a second pass so a module that is
  // only ever exercised is reported separately rather than as dead.
  const testedOnly = new Set();
  const walkTests = (dir) => {
    for (const entry of readdirSync(dir)) {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) {
        walkTests(path);
        continue;
      }
      if (!/\.test\.tsx?$/.test(entry)) continue;
      const file = path.replace(/\\/g, '/');
      for (const specifier of specifiersIn(readFileSync(file, 'utf8'))) {
        const target = resolveSpecifier(file, specifier);
        if (target && !reachable.has(target)) testedOnly.add(target);
      }
    }
  };
  walkTests(ROOT);

  const unreachable = modules.filter(
    (module) => !reachable.has(module) && !testedOnly.has(module),
  );

  if (unreachable.length > 0) {
    console.error('Unreachable frontend modules:\n');
    for (const module of unreachable) console.error(`  - ${module}`);
    console.error(
      `\n${unreachable.length} module(s) cannot be reached from ${ENTRY_POINTS.join(' or ')}, ` +
        'and no test imports them.\n' +
        'Wire each one up, or delete it. A module nothing reaches still gets read by the ' +
        'next person, and — if it calls a Tauri command — still makes that command look ' +
        'consumed.',
    );
    process.exit(1);
  }

  console.log(
    `Reachability OK: ${reachable.size} module(s) reachable from the entry points, ` +
      `${testedOnly.size} reached only by tests.`,
  );
}

main();
