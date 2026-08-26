import { getBackendService } from './api';

export type Role =
  | 'administrator'
  | 'modelAdministrator'
  | 'knowledgeAdministrator'
  | 'user'
  | 'reviewer'
  | 'auditor';

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

export interface User {
  id: string;
  displayName: string;
  roles: Role[];
  department: string | null;
}

export interface Session {
  user: User;
  startedAt: string;
}

export type AuditKind =
  | 'modeChanged'
  | 'egressDecision'
  | 'policyDecision'
  | 'session'
  | 'modelRegistry'
  | 'knowledge'
  | 'task'
  | 'approval';

/** One sealed record. `hash` covers both its contents and its position. */
export interface AuditEntry {
  seq: number;
  at: string;
  actor: string;
  kind: AuditKind;
  summary: string;
  detail: Record<string, unknown> | null;
  hash: string;
}

/** The result of recomputing every seal from the first entry onward. */
export interface ChainVerification {
  entriesChecked: number;
  intact: boolean;
  /** First entry whose seal disagrees. Everything before it remains verifiable. */
  firstBrokenSeq: number | null;
  detail: string;
}

/**
 * Whether this deployment has been set up.
 *
 * `awaitingFirstAdministrator` means no account has a password yet — the very
 * first thing that must happen is an administrator choosing one.
 */
export type AuthenticationStatus = 'awaitingFirstAdministrator' | 'configured';

export const ROLE_LABELS: Record<Role, string> = {
  administrator: 'Administrator',
  modelAdministrator: 'Model administrator',
  knowledgeAdministrator: 'Knowledge administrator',
  user: 'User',
  reviewer: 'Reviewer',
  auditor: 'Auditor',
};

export const AUDIT_KIND_LABELS: Record<AuditKind, string> = {
  modeChanged: 'Mode',
  egressDecision: 'Network',
  policyDecision: 'Policy',
  session: 'Session',
  modelRegistry: 'Models',
  knowledge: 'Knowledge',
  task: 'Task',
  approval: 'Approval',
};

export const governanceService = {
  listAccounts(): Promise<User[]> {
    return getBackendService().invoke<User[]>('list_accounts');
  },

  authenticationStatus(): Promise<AuthenticationStatus> {
    return getBackendService().invoke<AuthenticationStatus>('authentication_status');
  },

  /** Rejects with "That account and password do not match" on any failure. */
  signIn(userId: string, password: string): Promise<Session> {
    return getBackendService().invoke<Session>('sign_in', { userId, password });
  },

  /** Only available on a deployment where nobody has a password yet. */
  setInitialAdministratorPassword(userId: string, password: string): Promise<void> {
    return getBackendService().invoke<void>('set_initial_administrator_password', {
      userId,
      password,
    });
  },

  /** Administrators only. */
  setAccountPassword(userId: string, password: string): Promise<void> {
    return getBackendService().invoke<void>('set_account_password', { userId, password });
  },

  signOut(): Promise<void> {
    return getBackendService().invoke<void>('sign_out');
  },

  currentSession(): Promise<Session | null> {
    return getBackendService().invoke<Session | null>('current_session');
  },

  currentPermissions(): Promise<Permission[]> {
    return getBackendService().invoke<Permission[]>('current_permissions');
  },

  /** Rejects when the signed-in user may not read the record. */
  recentEntries(limit = 200): Promise<AuditEntry[]> {
    return getBackendService().invoke<AuditEntry[]>('recent_audit_entries', { limit });
  },

  /** Recomputes every seal. This is what turns the log into evidence. */
  verifyChain(): Promise<ChainVerification> {
    return getBackendService().invoke<ChainVerification>('verify_audit_chain');
  },
};
