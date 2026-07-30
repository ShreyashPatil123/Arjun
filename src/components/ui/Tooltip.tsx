import React, { useState } from 'react';
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
  let timeout: NodeJS.Timeout;

  const show = () => { timeout = setTimeout(() => setVisible(true), delay); };
  const hide = () => { clearTimeout(timeout); setVisible(false); };

  return (
    <div className={styles.container} onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      <div className={classNames(styles.tooltip, styles[position], visible && styles.visible)}>
        {content}
      </div>
    </div>
  );
};