import React from 'react';
import { classNames } from '../../utils/helpers';
import styles from './Badge.module.css';

interface BadgeProps {
  variant?: 'default' | 'success' | 'error' | 'warning' | 'info';
  children?: React.ReactNode;
  dot?: boolean;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ variant = 'default', children, dot, className }) => {
  return (
    <span className={classNames(styles.badge, styles[variant], dot && styles.dot, className)}>
      {!dot && children}
    </span>
  );
};