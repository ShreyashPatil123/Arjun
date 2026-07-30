import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { AppConfig } from '../types/config';
import { getSarathiClient } from '../sdk';

interface ConfigContextType {
  config: AppConfig | null;
  setConfig: (config: AppConfig) => void;
}

const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

export function ConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfigState] = useState<AppConfig | null>(null);

  useEffect(() => {
    getSarathiClient().config.getConfig().then(setConfigState).catch(() => {
      // Fallback
    });
  }, []);

  const setConfig = (newConfig: AppConfig) => {
    setConfigState(newConfig);
    getSarathiClient().config.setConfig(newConfig).catch(console.error);
  };

  return (
    <ConfigContext.Provider value={{ config, setConfig }}>
      {children}
    </ConfigContext.Provider>
  );
}

export function useConfig() {
  const context = useContext(ConfigContext);
  if (!context) throw new Error('useConfig must be used within ConfigProvider');
  return context;
}