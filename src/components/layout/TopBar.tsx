import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Sun, Moon, Settings, HardDrive, Cpu, Rocket, Library } from 'lucide-react';
import { SarathiLogo } from '../ui/SarathiLogo';
import { useTheme } from '../../hooks/useTheme';
import styles from './TopBar.module.css';

export const TopBar = () => {
  const { resolvedTheme, setTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const toggleTheme = () => {
    setTheme(resolvedTheme === 'dark' ? 'light' : 'dark');
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className={styles.topbar}>
      <div className={styles.left} style={{ cursor: 'pointer' }} onClick={() => navigate('/')}>
        <div className={styles.logo}>
          <SarathiLogo size={24} strokeWidth={2} title={null} />
        </div>
        <span className={styles.title}>Sarathi</span>
      </div>

      <nav style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        {/* Launch comes first: starting a tool is the main thing this app is for. */}
        <button
          className={styles.iconBtn}
          style={{
            display: 'flex',
            gap: '6px',
            fontSize: '13px',
            fontWeight: isActive('/launch') ? 600 : 400,
            color: isActive('/launch') ? 'var(--accent-primary)' : undefined,
            background: isActive('/launch') ? 'var(--bg-hover)' : undefined,
            padding: '6px 12px',
          }}
          onClick={() => navigate('/launch')}
        >
          <Rocket size={16} />
          Launch
        </button>

        <button
          className={styles.iconBtn}
          style={{
            display: 'flex',
            gap: '6px',
            fontSize: '13px',
            fontWeight: isActive('/browse') ? 600 : 400,
            color: isActive('/browse') ? 'var(--accent-primary)' : undefined,
            background: isActive('/browse') ? 'var(--bg-hover)' : undefined,
            padding: '6px 12px',
          }}
          onClick={() => navigate('/browse')}
        >
          <Library size={16} />
          Discover
        </button>

        <button
          className={styles.iconBtn}
          style={{
            display: 'flex',
            gap: '6px',
            fontSize: '13px',
            fontWeight: isActive('/system') ? 600 : 400,
            color: isActive('/system') ? 'var(--accent-primary)' : undefined,
            background: isActive('/system') ? 'var(--bg-hover)' : undefined,
            padding: '6px 12px',
          }}
          onClick={() => navigate('/system')}
        >
          <Cpu size={16} />
          System
        </button>

        <button
          className={styles.iconBtn}
          style={{
            display: 'flex',
            gap: '6px',
            fontSize: '13px',
            fontWeight: isActive('/models') ? 600 : 400,
            color: isActive('/models') ? 'var(--accent-primary)' : undefined,
            background: isActive('/models') ? 'var(--bg-hover)' : undefined,
            padding: '6px 12px',
          }}
          onClick={() => navigate('/models')}
        >
          <HardDrive size={16} />
          Storage
        </button>

      </nav>

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