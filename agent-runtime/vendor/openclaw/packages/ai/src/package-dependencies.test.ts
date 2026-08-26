// ARJUN sovereign build: replaces upstream's hardcoded dependency contract.
//
// Upstream asserted a fixed pair -- model-catalog-core and normalization-core --
// each declared as "workspace:*". Both halves are wrong here. Pruning the cloud
// providers removed the only production import of model-catalog-core and left
// media-core newly visible, and this workspace is installed by npm, which uses
// "*" rather than pnpm's "workspace:" protocol.
//
// Hardcoding a list is what made the original brittle, so this derives the set
// from the sources instead: whatever production code imports must be declared,
// and nothing declared may be missing from the vendored tree. That keeps the
// contract meaningful through the next prune or upstream re-sync.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const PACKAGE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC_ROOT = path.join(PACKAGE_ROOT, "src");
const PACKAGES_ROOT = path.resolve(PACKAGE_ROOT, "..");

/** npm's in-workspace version marker. pnpm would write "workspace:*". */
const WORKSPACE_MARKER = "*";

async function productionSourceFiles(dir: string, out: string[] = []): Promise<string[]> {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await productionSourceFiles(full, out);
    } else if (
      entry.name.endsWith(".ts") &&
      !entry.name.endsWith(".test.ts") &&
      !entry.name.includes("test-support") &&
      !entry.name.includes("test-helpers")
    ) {
      out.push(full);
    }
  }
  return out;
}

/** Sibling @openclaw packages imported by production code, excluding self. */
async function importedSiblingPackages(): Promise<Set<string>> {
  const found = new Set<string>();
  for (const file of await productionSourceFiles(SRC_ROOT)) {
    const text = await fs.readFile(file, "utf8");
    for (const match of text.matchAll(/from "(@openclaw\/[a-z0-9-]+)/g)) {
      const name = match[1];
      if (name && name !== "@openclaw/ai") found.add(name);
    }
  }
  return found;
}

async function readManifest(): Promise<{
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
}> {
  return JSON.parse(await fs.readFile(path.join(PACKAGE_ROOT, "package.json"), "utf8"));
}

describe("@openclaw/ai source dependency contract", () => {
  it("declares every sibling package its production sources import", async () => {
    const imported = [...(await importedSiblingPackages())].sort();
    const manifest = await readManifest();
    const declared = { ...manifest.dependencies, ...manifest.devDependencies };

    // Guards against the derivation silently matching nothing.
    expect(imported.length).toBeGreaterThan(0);

    for (const name of imported) {
      expect(
        declared[name],
        `${name} is imported by production code but not declared in package.json`,
      ).toBe(WORKSPACE_MARKER);
    }
  });

  it("keeps sibling packages out of runtime dependencies", async () => {
    // Workspace siblings are consumed as source through the tsconfig paths map,
    // not resolved at runtime, so they belong in devDependencies.
    const manifest = await readManifest();
    for (const name of Object.keys(manifest.dependencies ?? {})) {
      expect(name.startsWith("@openclaw/")).toBe(false);
    }
  });

  it("resolves every declared sibling to a vendored package", async () => {
    const manifest = await readManifest();
    for (const name of Object.keys(manifest.devDependencies ?? {})) {
      if (!name.startsWith("@openclaw/")) continue;
      const dir = path.join(PACKAGES_ROOT, name.slice("@openclaw/".length));
      await expect(
        fs.access(path.join(dir, "package.json")),
        `${name} is declared but not vendored`,
      ).resolves.toBeUndefined();
    }
  });
});
