import React from 'react';
import { Outlet } from 'react-router-dom';
import { TopBar } from './TopBar';
import { StatusBar } from './StatusBar';
import styles from './AppShell.module.css';

export const AppShell = () => {
  return (
    <div className={styles.shell}>
      <TopBar />
      <main className={styles.main}>
        <Outlet />
      </main>
      <StatusBar />
    </div>
  );
};