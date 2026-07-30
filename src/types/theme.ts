export type Theme = 'dark' | 'light' | 'system';

export interface ThemeConfig {
  current: Theme;
  resolved: 'dark' | 'light';
}