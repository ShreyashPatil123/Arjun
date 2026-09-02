import { getBackendService } from './api';
import { AppState } from '../types/app-state';
import { logActivity as dbLogActivity } from './database.service';

export async function getAppInfo(): Promise<{ version: string; name: string }> {
  return getBackendService().invoke<{ version: string; name: string }>('get_app_info');
}

export async function getAppState(): Promise<AppState> {
  // `get_app_state_info` is the command; `get_app_state()` is the Rust
  // function it calls internally. Naming the latter here invoked a command
  // that does not exist, and `AppStateContext` swallows the rejection — so
  // the whole app ran with no status, version or first-run flag, and nothing
  // said why.
  return getBackendService().invoke<AppState>('get_app_state_info');
}

export async function logActivity(action: string, category: string, details?: string): Promise<void> {
  return dbLogActivity(action, category, details || '');
}