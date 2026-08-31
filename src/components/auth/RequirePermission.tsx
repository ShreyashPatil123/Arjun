import React, { ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { usePermissions, type Permission } from '../../contexts/PermissionsContext';

/**
 * Route-level guard.
 *
 * Renders the children only if the signed-in user holds every
 * `permission` listed. Otherwise redirects to `/` (the workbench) and
 * lets the in-app toast surface the reason. The workbench is the safe
 * default: every signed-in role can reach it.
 *
 * The back-end remains the security boundary. This component prevents
 * a user from reaching a page that would surface a refusal toast on
 * load, not from a malicious caller who knows the route.
 */
export function RequirePermission({
  permission,
  children,
}: {
  /** The single permission required. (Multi-permission routes are not yet needed.) */
  permission: Permission;
  children: ReactNode;
}) {
  const { has, ready, session } = usePermissions();
  const location = useLocation();

  if (!ready) return null;

  // Not signed in — the App-level gate should already have routed
  // to /sign-in, but defence in depth.
  if (!session) {
    return <Navigate to="/sign-in" state={{ from: location }} replace />;
  }

  if (!has(permission)) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}
