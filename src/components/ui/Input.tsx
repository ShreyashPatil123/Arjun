import React from 'react';
import { classNames } from '../../utils/helpers';
import styles from './Input.module.css';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
  error?: boolean;
  icon?: React.ReactNode;
}

export const Input: React.FC<InputProps> = ({ label, helperText, error, icon, className, id, disabled, ...props }) => {
  return (
    <div className={classNames(styles.container, className)}>
      {label && <label htmlFor={id} className={styles.label}>{label}</label>}
      <div className={styles.inputWrapper}>
        {icon && <span className={styles.icon}>{icon}</span>}
        <input
          id={id}
          className={classNames(
            styles.input,
            icon && styles.hasIcon,
            error && styles.error,
            disabled && styles.disabled
          )}
          disabled={disabled}
          {...props}
        />
      </div>
      {helperText && (
        <span className={classNames(styles.helperText, error && styles.helperError)}>
          {helperText}
        </span>
      )}
    </div>
  );
};