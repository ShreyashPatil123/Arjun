import React, { createContext, useContext, useEffect, useState } from 'react';
import { Theme } from '../types/theme';
import { getSarathiClient } from '../sdk';

interface ThemeContextType {
  theme: Theme;
  resolvedTheme: 'dark' | 'light';
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('system');
  const [resolvedTheme, setResolvedTheme] = useState<'dark' | 'light'>('dark');

  useEffect(() => {
    const client = getSarathiClient();
    client.theme.getTheme().then(initial => {
      setThemeState(initial);
      const res = initial === 'system' ? client.theme.getSystemTheme() : initial;
      setResolvedTheme(res);
      client.theme.applyTheme(res);
    });

    const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
    const handleChange = (e: MediaQueryListEvent) => {
      if (theme === 'system') {
        const newTheme = e.matches ? 'light' : 'dark';
        setResolvedTheme(newTheme);
        client.theme.applyTheme(newTheme);
      }
    };
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  const setTheme = (newTheme: Theme) => {
    const client = getSarathiClient();
    setThemeState(newTheme);
    client.theme.setTheme(newTheme);
    const res = newTheme === 'system' ? client.theme.getSystemTheme() : newTheme;
    setResolvedTheme(res);
    client.theme.applyTheme(res);
  };

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
}