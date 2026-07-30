import React from 'react';
import { classNames } from '../../utils/helpers';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  hoverable?: boolean;
  padding?: 'sm' | 'md' | 'lg';
}

export const Card: React.FC<CardProps> = ({ children, header, footer, className, hoverable, padding = 'md' }) => {
  return (
    <div className={classNames(styles.card, hoverable && styles.hoverable, className)}>
      {header && <div className={styles.header}>{header}</div>}
      <div className={styles[`padding-${padding}`]}>{children}</div>
      {footer && <div className={styles.footer}>{footer}</div>}
    </div>
  );
};