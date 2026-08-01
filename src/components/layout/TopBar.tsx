import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Sun, Moon, Settings, Compass, HardDrive, MessageSquare, Cpu } from 'lucide-react';
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
          <Compass size={24} strokeWidth={2.5} />
        </div>
        <span className={styles.title}>Sarathi</span>
      </div>

      <nav style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
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
          Models & Storage
        </button>

        <button
          className={styles.iconBtn}
          style={{
            display: 'flex',
            gap: '6px',
            fontSize: '13px',
            fontWeight: isActive('/chat') ? 600 : 400,
            color: isActive('/chat') ? 'var(--accent-primary)' : undefined,
            background: isActive('/chat') ? 'var(--bg-hover)' : undefined,
            padding: '6px 12px',
          }}
          onClick={() => navigate('/chat')}
        >
          <MessageSquare size={16} />
          AI Chat
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