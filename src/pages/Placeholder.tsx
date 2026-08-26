import React from 'react';
import styles from './Placeholder.module.css';

interface PlaceholderProps {
  title: string;
  /** What this surface will do once its phase lands. */
  purpose: string;
  /** Build phase from the ARJUN plan, so the gap is legible rather than a dead end. */
  phase: string;
}

/** A surface that is routed and named but not yet implemented.
 *
 *  Deliberately states what is missing instead of showing mock content: a fake
 *  populated screen is indistinguishable from a working one during a demo, and
 *  that is exactly the confusion this project cannot afford. */
export const Placeholder = ({ title, purpose, phase }: PlaceholderProps) => (
  <div className={styles.page}>
    <div className={styles.card}>
      <h1 className={styles.title}>{title}</h1>
      <p className={styles.purpose}>{purpose}</p>
      <p className={styles.phase}>Not yet built &middot; {phase}</p>
    </div>
  </div>
);
