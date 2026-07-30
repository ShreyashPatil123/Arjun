import { ISarathiClient } from './ISarathiClient';
import * as configService from '../services/config.service';
import * as appService from '../services/app.service';
import * as dbService from '../services/database.service';
import * as themeService from '../services/theme.service';
import * as systemService from '../services/system.service';
import * as modelService from '../services/model.service';
import * as providerService from '../services/provider.service';
import * as aiService from '../services/ai.service';
import * as loraService from '../services/lora.service';
import { AppConfig } from '../types/config';
import { Theme } from '../types/theme';

export class SarathiTauriClient implements ISarathiClient {
  readonly config = {
    getConfig: () => configService.getConfig(),
    setConfig: (config: AppConfig) => configService.setConfig(config),
    getConfigValue: <T = unknown>(key: string) => configService.getConfigValue<T>(key),
    setConfigValue: (key: string, value: unknown) => configService.setConfigValue(key, value),
    getAppPaths: () => configService.getAppPaths(),
    resetConfig: () => configService.resetConfig(),
  };

  readonly system = {
    getAppInfo: () => appService.getAppInfo(),
    getAppState: () => appService.getAppState(),
    logActivity: (action: string, category: string, details?: string) => appService.logActivity(action, category, details),
  };

  readonly database = {
    getSetting: (key: string) => dbService.getSetting(key),
    setSetting: (key: string, value: string, type?: string) => dbService.setSetting(key, value, type || 'string'),
    getAllSettings: () => dbService.getAllSettings(),
    getRecentActivity: (limit?: number) => dbService.getRecentActivity(limit || 10),
  };

  readonly theme = {
    getTheme: () => themeService.getTheme(),
    setTheme: (theme: Theme) => themeService.setTheme(theme),
    getSystemTheme: () => themeService.getSystemTheme(),
    applyTheme: (theme: 'dark' | 'light') => themeService.applyTheme(theme),
  };

  // Phase 2: System Analyzer
  readonly systemAnalyzer = {
    getHardwareProfile: async () => systemService.getHardwareProfile(),
  };

  // Phase 3: Model Manager
  readonly modelManager = {
    listModels: async () => modelService.listModels(),
    getModelCompatibility: async (id: string) => modelService.getModelCompatibility(id),
    getRecommendations: async () => modelService.getRecommendations(),
  };

  // Phase 4: Model Providers
  readonly modelProviders = {
    getProviders: async () => providerService.getProviders(),
    searchProviderModels: async (pId: string, q: string) => providerService.searchProviderModels(pId, q),
  };

  // Phase 5: AI Engine
  readonly aiEngine = {
    loadModel: async (path: string) => aiService.loadModel(path),
    unloadModel: async () => aiService.unloadModel(),
    chat: async (msgs?: unknown[]) => aiService.chat(msgs),
  };

  // Phase 6: Dynamic LoRA Orchestration
  readonly lora = {
    loadAdapter: async (id: string) => loraService.loadAdapter(id),
    switchAdapter: async (id: string) => loraService.switchAdapter(id),
    composeAdapters: async (adapters: unknown[]) => loraService.composeAdapters(adapters),
  };
}
