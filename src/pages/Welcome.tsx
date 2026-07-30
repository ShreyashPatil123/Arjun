import React from 'react';
import { Compass } from 'lucide-react';
import styles from './Welcome.module.css';

export const Welcome = () => {
  return (
    <div className={styles.container}>
      <div className={styles.orb} />
      <div className={styles.logoWrapper}>
        <Compass size={64} strokeWidth={2} />
      </div>
      <h1 className={styles.title}>Welcome to Sarathi</h1>
      <p className={styles.subtitle}>A Local-First LoRA Orchestration System</p>
      
      <div className={styles.pills}>
        <span className={styles.pill}>Local First</span>
        <span className={styles.pill}>Privacy First</span>
        <span className={styles.pill}>Modular</span>
      </div>

      <div className={styles.hint}>Phase 2: System Analysis coming soon</div>
    </div>
  );
};