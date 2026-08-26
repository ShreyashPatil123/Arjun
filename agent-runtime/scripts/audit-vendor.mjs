/**
 * Vendor audit for the sovereign agent runtime.
 *
 * Two questions, both of which must stay answered "yes" for the sovereignty
 * claim to hold, and neither of which a human will reliably re-check by eye
 * after a re-sync with upstream OpenClaw:
 *
 *   1. Link integrity - does every relative import in the vendored tree still
 *      resolve? Pruning provider files by hand is exactly the operation that
 *      leaves a dangling `export * from "./anthropic.js"` behind.
 *   2. Provider absence - is any excluded cloud SDK, provider entrypoint or
 *      egress-capable transport back in the tree? A re-sync reintroduces them
 *      silently.
 *
 * Run: node scripts/audit-vendor.mjs
 */
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join, dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const VENDOR = join(ROOT, "vendor", "openclaw");

const BACKSLASH = String.fromCharCode(92);
/** Report paths the way a reader will type them, on any platform. */
const posix = (p) => relative(ROOT, p).split(BACKSLASH).join("/");

/** SDKs whose presence would mean a vendor endpoint is reachable from here. */
const FORBIDDEN_SPECIFIERS = [
  "@anthropic-ai/sdk",
  "@google/genai",
  "@mistralai/mistralai",
];

/** Provider entrypoints deliberately not vendored. */
const FORBIDDEN_MODULES = [
  "providers/anthropic.js",
  "providers/google.js",
  "providers/google-vertex.js",
  "providers/google-shared.js",
  "providers/mistral.js",
];

/** Only this API may be registered. Anything else reaches a vendor. */
const ALLOWED_REGISTERED_APIS = new Set(["openai-completions"]);

function walk(dir, filter, out = []) {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === ".git" || entry === "dist") continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, filter, out);
    else if (filter(entry)) out.push(full);
  }
  return out;
}

const sourceFiles = walk(VENDOR, (n) => /\.(ts|mts|js|mjs)$/.test(n));
const manifestFiles = walk(VENDOR, (n) => n === "package.json");
const problems = [];

// --- 1. Link integrity ------------------------------------------------------
// TypeScript ESM writes ".js" in the specifier for a ".ts" source, so probe both.
const CANDIDATES = ["", ".ts", ".mts", ".tsx", "/index.ts", "/index.mts"];

function resolvesOnDisk(fromFile, spec) {
  const base = resolve(dirname(fromFile), spec);
  const stripped = base.replace(/\.(js|mjs)$/, "");
  return CANDIDATES.some((s) => existsSync(stripped + s) || existsSync(base + s));
}

const IMPORT_RE = /(?:from|import)\s*\(?\s*["'](\.[^"']+)["']/g;

for (const file of sourceFiles) {
  const text = readFileSync(file, "utf8");
  for (const m of text.matchAll(IMPORT_RE)) {
    if (!resolvesOnDisk(file, m[1])) {
      problems.push(`dangling import   ${posix(file)} -> ${m[1]}`);
    }
  }
}

// --- 2. Provider absence ----------------------------------------------------
for (const file of sourceFiles) {
  const text = readFileSync(file, "utf8");
  for (const spec of FORBIDDEN_SPECIFIERS) {
    // Match real imports only, so a comment explaining the removal does not trip.
    const escaped = spec.replace(/[/@.\-]/g, (c) => BACKSLASH + c);
    if (new RegExp(`(?:from|import)\\s*\\(?\\s*["']${escaped}`).test(text)) {
      problems.push(`forbidden SDK     ${posix(file)} imports ${spec}`);
    }
  }
  for (const mod of FORBIDDEN_MODULES) {
    if (text.includes(`"./${mod}"`) || text.includes(`"../${mod}"`)) {
      problems.push(`excluded provider ${posix(file)} references ${mod}`);
    }
  }
}

// --- 3. Registry contains only the local transport --------------------------
const registerBuiltins = join(VENDOR, "packages/ai/src/providers/register-builtins.ts");
if (existsSync(registerBuiltins)) {
  const text = readFileSync(registerBuiltins, "utf8");
  const at = text.indexOf("const registerBuiltIns");
  const body = at === -1 ? "" : text.slice(at);
  const registered = [...body.matchAll(/createLazyRegistration\(\s*"([^"]+)"/g)].map((m) => m[1]);
  for (const api of registered) {
    if (!ALLOWED_REGISTERED_APIS.has(api)) {
      problems.push(`registered API    "${api}" reaches a vendor endpoint`);
    }
  }
  if (registered.length === 0) {
    problems.push("registered API    none found - the registry patch may have been lost");
  }
  console.log(`registered APIs: ${registered.join(", ") || "(none)"}`);
} else {
  problems.push("register-builtins.ts is missing from the vendored tree");
}

// --- 4. Declared dependencies -----------------------------------------------
for (const pj of manifestFiles) {
  const d = JSON.parse(readFileSync(pj, "utf8"));
  for (const field of ["dependencies", "devDependencies", "peerDependencies"]) {
    for (const dep of Object.keys(d[field] ?? {})) {
      if (FORBIDDEN_SPECIFIERS.includes(dep)) {
        problems.push(`forbidden dep     ${posix(pj)} declares ${dep}`);
      }
    }
  }
}

console.log(`scanned ${sourceFiles.length} source files and ${manifestFiles.length} manifests`);
if (problems.length > 0) {
  console.error(`\nFAIL - ${problems.length} problem(s):`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log("OK - link integrity and provider exclusion both hold");
