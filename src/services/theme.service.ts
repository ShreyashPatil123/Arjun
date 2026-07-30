import { Theme } from '../types/theme';
import { getSetting, setSetting } from './database.service';

export async function getTheme(): Promise<Theme> {
  const setting = await getSetting('theme');
  return (setting?.value as Theme) || 'system';
}

export async function setTheme(theme: Theme): Promise<void> {
  await setSetting('theme', theme, 'string');
}

export function getSystemTheme(): 'dark' | 'light' {
  if (typeof window === 'undefined') return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function applyTheme(theme: Theme): void {
  const resolved = theme === 'system' ? getSystemTheme() : theme;
  document.documentElement.setAttribute('data-theme', resolved);
}