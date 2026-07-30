import { invoke } from '@tauri-apps/api/core';

export interface IBackendService {
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
}

class TauriBackendService implements IBackendService {
  async invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
    return invoke<T>(command, args);
  }
}

let backendService: IBackendService = new TauriBackendService();

export function getBackendService(): IBackendService {
  return backendService;
}

export function setBackendService(service: IBackendService): void {
  backendService = service;
}