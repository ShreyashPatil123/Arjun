import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Sun, Moon, Settings, Compass } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import styles from './TopBar.module.css';

export const TopBar = () => {
  const { resolvedTheme, setTheme } = useTheme();
  const navigate = useNavigate();

  const toggleTheme = () => {
    setTheme(resolvedTheme === 'dark' ? 'light' : 'dark');
  };

  return (
    <header className={styles.topbar}>
      <div className={styles.left} style={{ cursor: 'pointer' }} onClick={() => navigate('/')}>
        <div className={styles.logo}>
          <Compass size={24} strokeWidth={2.5} />
        </div>
        <span className={styles.title}>Sarathi</span>
      </div>
      <div className={styles.right}>
        <button className={styles.iconBtn} onClick={toggleTheme} aria-label="Toggle theme">
          {resolvedTheme === 'dark' ? <Sun size={20} className={styles.animateIcon} /> : <Moon size={20} className={styles.animateIcon} />}
        </button>
        <button className={styles.iconBtn} onClick={() => navigate('/settings')} aria-label="Settings">
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
};