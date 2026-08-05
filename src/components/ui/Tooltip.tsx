import React, { useRef, useState } from 'react';
import { classNames } from '../../utils/helpers';
import styles from './Tooltip.module.css';

interface TooltipProps {
  content: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  children: React.ReactNode;
  delay?: number;
}

export const Tooltip: React.FC<TooltipProps> = ({ content, position = 'top', children, delay = 200 }) => {
  const [visible, setVisible] = useState(false);
  // `ReturnType<typeof setTimeout>` avoids depending on @types/node, which is
  // not installed — `NodeJS.Timeout` failed to resolve and broke `npm run build`.
  // A ref rather than a local: a plain `let` is reinitialised on every render,
  // so `hide()` could clear a stale handle and leave the tooltip stuck open.
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = () => { timeoutRef.current = setTimeout(() => setVisible(true), delay); };
  const hide = () => {
    if (timeoutRef.current !== null) clearTimeout(timeoutRef.current);
    setVisible(false);
  };

  return (
    <div className={styles.container} onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      <div className={classNames(styles.tooltip, styles[position], visible && styles.visible)}>
        {content}
      </div>
    </div>
  );
};