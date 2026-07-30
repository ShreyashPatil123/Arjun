import React from 'react';
import { useAppState } from '../../hooks/useAppState';
import { classNames } from '../../utils/helpers';
import styles from './StatusBar.module.css';

export const StatusBar = () => {
  const { state } = useAppState();
  const status = state?.status || 'initializing';

  return (
    <footer className={styles.statusbar}>
      <div>v{state?.version || '0.1.0'}</div>
      <div className={styles.center}>
        <span className={classNames(styles.dot, styles[status === 'ready' ? 'ready' : 'initializing'])} />
        <span style={{ textTransform: 'capitalize' }}>{status}</span>
      </div>
      <div></div>
    </footer>
  );
};