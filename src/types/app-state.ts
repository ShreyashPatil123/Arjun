export type AppStatus = 
  | 'initializing'
  | 'ready'
  | 'downloading'
  | 'installing'
  | 'loading-model'
  | 'loading-lora'
  | 'chatting'
  | 'error';

export interface AppState {
  status: AppStatus;
  version: string;
  isFirstRun: boolean;
}