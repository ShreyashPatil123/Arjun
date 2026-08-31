import React, { ReactNode } from 'react';
import { useHas, type Permission } from '../../contexts/PermissionsContext';

/**
 * Inline gate for a button, link, or other interactive control.
 *
 * Renders `children` only if the signed-in user holds `permission`.
 * Otherwise renders `fallback` (default: nothing).
 *
 * The back-end is still the security boundary; this is a UX
 * primitive. A caller that calls the underlying Tauri command
 * directly will still be refused.
 */
export function Can({
  permission,
  fallback = null,
  children,
}: {
  permission: Permission;
  fallback?: ReactNode;
  children: ReactNode;
}) {
  const has = useHas(permission);
  return <>{has ? children : fallback}</>;
}

/**
 * Disabled-button variant. Renders the children either way, but
 * adds `aria-disabled` and `disabled` and a `data-rbac="denied"`
 * attribute the global stylesheet can hook into.
 *
 * Useful when the button is the only affordance in a row and a
 * "missing" element would confuse the layout.
 */
export function Cant({
  permission,
  reason,
  children,
  className,
  ...rest
}: {
  permission: Permission;
  /** Human-readable explanation shown in a tooltip when the button is disabled. */
  reason?: string;
  children: ReactNode;
  className?: string;
} & Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'children' | 'disabled'>) {
  const has = useHas(permission);
  return (
    <button
      type="button"
      className={className}
      disabled={!has}
      aria-disabled={!has}
      data-rbac={has ? undefined : 'denied'}
      title={!has ? (reason ?? `This action requires the ${permission} permission.`) : undefined}
      {...rest}
    >
      {children}
    </button>
  );
}
