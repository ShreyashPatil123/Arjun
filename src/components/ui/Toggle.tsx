import React from 'react';
import { classNames } from '../../utils/helpers';
import styles from './Toggle.module.css';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  size?: 'sm' | 'md';
  disabled?: boolean;
}

export const Toggle: React.FC<ToggleProps> = ({ checked, onChange, label, size = 'md', disabled }) => {
  return (
    <label className={classNames(styles.container, disabled && styles.disabled)}>
      <div
        className={classNames(styles.toggle, styles[size], checked && styles.checked)}
        onClick={() => !disabled && onChange(!checked)}
        role="switch"
        aria-checked={checked}
      >
        <div className={styles.thumb} />
      </div>
      {label && <span className={styles.label}>{label}</span>}
    </label>
  );
};