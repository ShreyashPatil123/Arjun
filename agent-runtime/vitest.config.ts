import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vitest/config";

const ROOT = dirname(fileURLToPath(import.meta.url));
const VENDOR = resolve(ROOT, "vendor/openclaw");

/**
 * Resolve `@openclaw/*` to vendored TypeScript source.
 *
 * The vendored manifests point `main`/`exports` at `dist/`, which upstream
 * produces with tsdown. We deliberately do not vendor a build step: the source
 * is what gets reviewed for the sovereignty claim, and a prebuilt `dist` would
 * be an unreviewed artifact sitting in the audit path. So tests and the runtime
 * consume `src/` directly.
 *
 * The mapping comes from `vendor/openclaw/tsconfig.json`'s `paths`, not from
 * the packages' `exports` maps and not from file layout. Subpath, export target
 * and source layout genuinely disagree upstream -- `@openclaw/ai/event-stream`
 * is declared as `dist/event-stream.mjs` but lives at `src/utils/event-stream.ts`,
 * because the build flattens directories. `paths` is the one table that already
 * records the true source location, and using it here means tsc and vitest
 * cannot drift apart.
 */
type PathsMap = Record<string, string[]>;

function loadPaths(): PathsMap {
  const raw = readFileSync(join(VENDOR, "tsconfig.json"), "utf8");
  // The config carries a "//" comment array, which JSON.parse handles fine.
  return (JSON.parse(raw).compilerOptions?.paths ?? {}) as PathsMap;
}

const PATHS = loadPaths();
const EXTENSIONS = ["", ".ts", ".mts", ".tsx", "/index.ts", "/index.mts"];

function firstExisting(candidate: string): string | undefined {
  const stripped = candidate.replace(/\.(js|mjs)$/, "");
  for (const ext of EXTENSIONS) {
    if (existsSync(stripped + ext)) return stripped + ext;
    if (existsSync(candidate + ext)) return candidate + ext;
  }
  return undefined;
}

function resolveFromPaths(id: string): string | undefined {
  // Exact entries win over wildcards, matching TypeScript's own precedence.
  const exact = PATHS[id];
  if (exact) {
    for (const target of exact) {
      const hit = firstExisting(join(VENDOR, target));
      if (hit) return hit;
    }
  }
  for (const [pattern, targets] of Object.entries(PATHS)) {
    if (!pattern.endsWith("/*")) continue;
    const prefix = pattern.slice(0, -1);
    if (!id.startsWith(prefix)) continue;
    const rest = id.slice(prefix.length);
    for (const target of targets) {
      const hit = firstExisting(join(VENDOR, target.replace("*", rest)));
      if (hit) return hit;
    }
  }
  return undefined;
}

function openclawSourceResolver(): Plugin {
  return {
    name: "arjun:openclaw-source-resolver",
    enforce: "pre",
    resolveId(id) {
      if (!id.startsWith("@openclaw/")) return null;
      return resolveFromPaths(id) ?? null;
    },
  };
}

export default defineConfig({
  plugins: [openclawSourceResolver()],
  test: {
    // Vendored upstream tests plus ARJUN's own runtime tests.
    include: ["vendor/openclaw/packages/**/*.test.ts", "src/**/*.test.ts"],
    exclude: ["**/node_modules/**", "**/dist/**"],
    passWithNoTests: true,
    // One vendored perf test streams 110KB through the reasoning-tag partitioner.
    testTimeout: 30_000,
  },
});
