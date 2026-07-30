import { ISarathiClient } from './ISarathiClient';
import { SarathiTauriClient } from './SarathiTauriClient';

export * from './ISarathiClient';
export * from './SarathiTauriClient';

let activeClient: ISarathiClient = new SarathiTauriClient();

/**
 * Returns the global Sarathi client instance.
 * Any UI component calls `getSarathiClient()` to interact with backend services.
 */
export function getSarathiClient(): ISarathiClient {
  return activeClient;
}

/**
 * Swaps the global Sarathi client implementation (e.g. for testing or alternative runtimes).
 */
export function setSarathiClient(client: ISarathiClient): void {
  activeClient = client;
}
