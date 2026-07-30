export interface AppConfig {
  theme: 'dark' | 'light' | 'system';
  language: string;
  backendUrl: string;
  ollamaUrl: string;
  modelDirectory: string;
  downloadDirectory: string;
  cacheDirectory: string;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  ai: AIConfig;
}

export interface AIConfig {
  defaultModel: string | null;
  contextLength: number;
  temperature: number;
  maxTokens: number;
}

export interface AppPaths {
  dataDir: string;
  logDir: string;
  configPath: string;
}