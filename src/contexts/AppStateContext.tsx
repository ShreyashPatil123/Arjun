import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { AppState, AppStatus } from '../types/app-state';
import { getSarathiClient } from '../sdk';

interface AppStateContextType {
  state: AppState | null;
  setStatus: (status: AppStatus) => void;
}

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState | null>(null);

  useEffect(() => {
    getSarathiClient().system.getAppState().then(setState).catch(() => {
      setState({ status: 'ready', version: '0.1.0', isFirstRun: false });
    });
  }, []);

  const setStatus = (status: AppStatus) => {
    setState(prev => prev ? { ...prev, status } : null);
  };

  return (
    <AppStateContext.Provider value={{ state, setStatus }}>
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) throw new Error('useAppState must be used within AppStateProvider');
  return context;
}