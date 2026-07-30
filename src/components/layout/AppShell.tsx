import React from 'react';
import { Outlet } from 'react-router-dom';
import { TopBar } from './TopBar';
import { StatusBar } from './StatusBar';
import { ErrorBoundary } from '../ui';
import styles from './AppShell.module.css';

export const AppShell = () => {
  return (
    <div className={styles.shell}>
      <TopBar />
      <main className={styles.main}>
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
      <StatusBar />
    </div>
  );
};