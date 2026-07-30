import { getBackendService } from './api';
import { AppState } from '../types/app-state';
import { logActivity as dbLogActivity } from './database.service';

export async function getAppInfo(): Promise<{ version: string; name: string }> {
  return getBackendService().invoke<{ version: string; name: string }>('get_app_info');
}

export async function getAppState(): Promise<AppState> {
  return getBackendService().invoke<AppState>('get_app_state');
}

export async function logActivity(action: string, category: string, details?: string): Promise<void> {
  return dbLogActivity(action, category, details || '');
}