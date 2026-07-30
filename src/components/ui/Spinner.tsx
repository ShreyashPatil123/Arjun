import React from 'react';
import styles from './Spinner.module.css';
import { Loader2 } from 'lucide-react';
import { classNames } from '../../utils/helpers';

export const Spinner = ({ size = 'md', color = 'var(--accent-primary)' }) => {
  return <Loader2 className={classNames(styles.spinner, styles[size])} color={color} />;
};