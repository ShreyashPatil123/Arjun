import { getBackendService } from './api';
import { AppConfig, AppPaths } from '../types/config';

export async function getConfig(): Promise<AppConfig> {
  return getBackendService().invoke<AppConfig>('get_config');
}

export async function setConfig(config: AppConfig): Promise<void> {
  return getBackendService().invoke<void>('set_config', { config });
}

export async function getConfigValue<T>(key: string): Promise<T> {
  return getBackendService().invoke<T>('get_config_value', { key });
}

export async function setConfigValue(key: string, value: unknown): Promise<void> {
  return getBackendService().invoke<void>('set_config_value', { key, value });
}

export async function getAppPaths(): Promise<AppPaths> {
  return getBackendService().invoke<AppPaths>('get_app_paths');
}

export async function resetConfig(): Promise<void> {
  return getBackendService().invoke<void>('reset_config');
}
/**
 * Whether HuggingFace requests are authenticated, and by what.
 *
 * The token itself is never returned — only whether one is set, so the Settings
 * field can say "configured" without echoing a secret back into the page.
 */
export interface HfTokenStatus {
  configured: boolean;
  /** `'settings'`, `'environment'`, or `'none'`. */
  source: 'settings' | 'environment' | 'none';
}

export async function getHfTokenStatus(): Promise<HfTokenStatus> {
  return getBackendService().invoke<HfTokenStatus>('get_hf_token_status');
}

/** Saves the token; an empty string clears it. Also drops the browse cache. */
export async function setHfToken(token: string): Promise<HfTokenStatus> {
  return getBackendService().invoke<HfTokenStatus>('set_hf_token', { token });
}
