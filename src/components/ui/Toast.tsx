import React from 'react';
import { CheckCircle, AlertTriangle, XCircle, Info, X } from 'lucide-react';
import { classNames } from '../../utils/helpers';
import styles from './Toast.module.css';
import { ToastMessage } from '../../contexts/ToastContext';

const icons = {
  success: <CheckCircle className={classNames(styles.icon, styles.success)} size={20} />,
  error: <XCircle className={classNames(styles.icon, styles.error)} size={20} />,
  warning: <AlertTriangle className={classNames(styles.icon, styles.warning)} size={20} />,
  info: <Info className={classNames(styles.icon, styles.info)} size={20} />
};

export const ToastContainer = ({ toasts, onClose }: { toasts: ToastMessage[], onClose: (id: string) => void }) => {
  return (
    <div className={styles.container}>
      {toasts.map(toast => (
        <div key={toast.id} className={styles.toast}>
          {icons[toast.type]}
          <div className={styles.content}>{toast.message}</div>
          <button className={styles.closeBtn} onClick={() => onClose(toast.id)}><X size={16} /></button>
          {toast.duration && toast.duration > 0 && (
            <div className={styles.progress} style={{ animationDuration: `${toast.duration}ms` }} />
          )}
        </div>
      ))}
    </div>
  );
};