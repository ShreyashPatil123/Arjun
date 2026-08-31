/**
 * Tests for the frontend's own logic.
 *
 * Deliberately narrow: `src/**` only, and only the pure modules. Rendering
 * tests would need a DOM implementation this repository does not vendor, and
 * the frontend logic actually worth testing — reconstructing a run from its
 * durable record — is pure by design so that it can be tested without one.
 *
 * Plain JavaScript rather than `defineConfig` in TypeScript, and run through
 * the copy of vitest already installed under `agent-runtime/`: a second copy at
 * the root would mean a network install, and this repository is built and
 * verified offline. Importing `defineConfig` here would need that second copy
 * for the config file alone.
 */
export default {
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'node',
  },
};
