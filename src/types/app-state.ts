export type AppStatus = 
  | 'initializing'
  | 'ready'
  | 'downloading'
  | 'installing'
  | 'loading-model'
  | 'chatting'
  | 'error';

export interface AppState {
  status: AppStatus;
  version: string;
  isFirstRun: boolean;
}
