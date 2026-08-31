import React, { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from 'react';
import { governanceService, type Session } from '../services/governance.service';

/**
 * The Permission values used by the matrix. These are the camelCase
 * strings returned by the back-end `current_permissions` command,
 * which the `serde(rename_all = "camelCase")` on the Rust enum maps
 * one-to-one.
 */
export type Permission =
  | 'useModel'
  | 'uploadDocument'
  | 'searchKnowledge'
  | 'executeCode'
  | 'writeFiles'
  | 'generateArtifact'
  | 'approveOutput'
  | 'importModel'
  | 'viewAuditLog'
  | 'modifyPolicy'
  | 'enterProvisioning';

interface PermissionsContextValue {
  session: Session | null;
  /** Every permission the signed-in user holds. Empty when nobody is signed in. */
  held: ReadonlySet<Permission>;
  /** True once the first read from the back-end has completed (success or failure). */
  ready: boolean;
  /** True if the user holds the given permission. */
  has: (permission: Permission) => boolean;
  /** Manually re-read (e.g. after sign-in or sign-out from a child). */
  refresh: () => Promise<void>;
}

const PermissionsContext = createContext<PermissionsContextValue | undefined>(undefined);

/**
 * Provides the session's permission set to the whole app.
 *
 * The matrix lives in Rust (`Role::grants(permission)`); this hook is
 * a thin React view of it. It reads once on mount and re-reads
 * whenever the session changes (e.g. sign-in / sign-out from a child).
 *
 * The back-end remains the security boundary. This hook is a UX
 * layer that lets the UI hide what the back-end would refuse, so the
 * user does not see a button that produces a refusal toast.
 */
export function PermissionsProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [held, setHeld] = useState<ReadonlySet<Permission>>(new Set());
  const [ready, setReady] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const next = await governanceService.currentSession();
      setSession(next);
      if (!next) {
        setHeld(new Set());
        return;
      }
      const granted = await governanceService.currentPermissions();
      setHeld(new Set((granted as Permission[]) ?? []));
    } catch {
      // The back-end may be unreachable (cold start, dev run without
      // the Tauri host). The UI must still render, and any call into
      // `has` will return false, which is the safe default.
      setSession(null);
      setHeld(new Set());
    } finally {
      setReady(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const has = useCallback(
    (permission: Permission) => held.has(permission),
    [held],
  );

  const value = useMemo<PermissionsContextValue>(
    () => ({ session, held, ready, has, refresh }),
    [session, held, ready, has, refresh],
  );

  return (
    <PermissionsContext.Provider value={value}>
      {children}
    </PermissionsContext.Provider>
  );
}

export function usePermissions(): PermissionsContextValue {
  const ctx = useContext(PermissionsContext);
  if (!ctx) {
    throw new Error('usePermissions must be used within PermissionsProvider');
  }
  return ctx;
}

/**
 * Sugar for the most common use: `const canManageModels = useHas('importModel')`.
 */
export function useHas(permission: Permission): boolean {
  return usePermissions().has(permission);
}
