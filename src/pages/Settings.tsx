import React from 'react';
import { Card, Toggle, Input, Button } from '../components/ui';
import { useTheme } from '../hooks/useTheme';
import { useConfig } from '../hooks/useConfig';
import styles from './Settings.module.css';

export const Settings = () => {
  const { resolvedTheme, setTheme } = useTheme();
  const { config } = useConfig();

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Settings</h1>
      
      <Card>
        <h2 className={styles.sectionTitle}>Appearance</h2>
        <div className={styles.row}>
          <div className={styles.info}>
            <span className={styles.label}>Dark Theme</span>
            <span className={styles.desc}>Toggle dark mode on or off.</span>
          </div>
          <Toggle 
            checked={resolvedTheme === 'dark'} 
            onChange={(checked) => setTheme(checked ? 'dark' : 'light')} 
          />
        </div>
      </Card>

      <Card>
        <h2 className={styles.sectionTitle}>Directories</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Input label="Model Directory" value={config?.modelDirectory || ''} readOnly disabled />
          <Input label="Download Directory" value={config?.downloadDirectory || ''} readOnly disabled />
          <Input label="Cache Directory" value={config?.cacheDirectory || ''} readOnly disabled />
        </div>
      </Card>

      <Card>
        <h2 className={styles.sectionTitle}>About</h2>
        <div className={styles.row}>
          <div className={styles.info}>
            <span className={styles.label}>Sarathi Version</span>
            <span className={styles.desc}>v0.1.0 (Phase 1)</span>
          </div>
          <Button variant="secondary">Check for Updates</Button>
        </div>
      </Card>
    </div>
  );
};