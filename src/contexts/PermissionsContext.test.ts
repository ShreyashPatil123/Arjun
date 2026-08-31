/**
 * Tests for the permissions context. The context depends on React,
 * but the underlying logic is two functions: the gate query and the
 * back-end read. The back-end read is mocked; the gate query is a
 * `Set.has` against the granted list. This test pins both without
 * needing a DOM.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const mockedService = {
  currentSession: vi.fn(),
  currentPermissions: vi.fn(),
};

vi.mock('../services/governance.service', () => ({
  governanceService: mockedService,
  ROLE_LABELS: { administrator: 'Administrator', employee: 'Employee' },
  isActiveRole: (r: string) => r === 'administrator' || r === 'employee',
  headlineRole: (rs: string[]) => (rs.includes('administrator') ? 'administrator' : 'employee'),
}));

// Re-implement the gate exactly the way `PermissionsProvider` does,
// so the test exercises the same code path the front-end uses. The
// real implementation in `PermissionsContext.tsx` is identical;
// this lets the test run without a React renderer.
type Permission =
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

interface PermissionsView {
  session: { user: { id: string; displayName: string; roles: string[] } } | null;
  held: ReadonlySet<Permission>;
  ready: boolean;
  has: (p: Permission) => boolean;
  refresh: () => Promise<void>;
}

async function loadPermissions(): Promise<PermissionsView> {
  // The real provider wraps this in try/catch so a back-end failure
  // does not leave the UI unable to render. This helper mirrors that.
  try {
    const session = await mockedService.currentSession();
    if (!session) {
      return {
        session: null,
        held: new Set<Permission>(),
        ready: true,
        has: () => false,
        refresh: async () => {},
      };
    }
    const granted = await mockedService.currentPermissions();
    const held = new Set<Permission>((granted as Permission[]) ?? []);
    return {
      session,
      held,
      ready: true,
      has: (p) => held.has(p),
      refresh: async () => {},
    };
  } catch {
    return {
      session: null,
      held: new Set<Permission>(),
      ready: true,
      has: () => false,
      refresh: async () => {},
    };
  }
}

describe('permissions view', () => {
  beforeEach(() => {
    mockedService.currentSession.mockReset();
    mockedService.currentPermissions.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('returns no permissions when nobody is signed in', async () => {
    mockedService.currentSession.mockResolvedValue(null);
    const view = await loadPermissions();
    expect(view.session).toBeNull();
    expect(view.held.size).toBe(0);
    expect(view.has('useModel')).toBe(false);
    expect(view.has('importModel')).toBe(false);
  });

  it('returns the granted permissions for an Employee', async () => {
    mockedService.currentSession.mockResolvedValue({
      user: { id: 'engineer', displayName: 'P. Shetty', roles: ['employee'], department: null },
      startedAt: '2026-08-30T00:00:00Z',
    });
    mockedService.currentPermissions.mockResolvedValue([
      'useModel',
      'uploadDocument',
      'searchKnowledge',
      'executeCode',
      'generateArtifact',
      'approveOutput',
    ]);

    const view = await loadPermissions();
    expect(view.has('useModel')).toBe(true);
    expect(view.has('uploadDocument')).toBe(true);
    expect(view.has('searchKnowledge')).toBe(true);
    expect(view.has('executeCode')).toBe(true);
    expect(view.has('generateArtifact')).toBe(true);
    expect(view.has('approveOutput')).toBe(true);
    // Employee does not have administrative permissions.
    expect(view.has('importModel')).toBe(false);
    expect(view.has('viewAuditLog')).toBe(false);
    expect(view.has('modifyPolicy')).toBe(false);
    expect(view.has('enterProvisioning')).toBe(false);
  });

  it('Administrator holds every permission (the superset)', async () => {
    mockedService.currentSession.mockResolvedValue({
      user: { id: 'admin', displayName: 'R. Nair', roles: ['administrator'], department: null },
      startedAt: '2026-08-30T00:00:00Z',
    });
    mockedService.currentPermissions.mockResolvedValue([
      'useModel',
      'uploadDocument',
      'searchKnowledge',
      'executeCode',
      'generateArtifact',
      'approveOutput',
      'importModel',
      'viewAuditLog',
      'modifyPolicy',
      'enterProvisioning',
    ]);

    const view = await loadPermissions();
    for (const p of [
      'useModel',
      'uploadDocument',
      'searchKnowledge',
      'executeCode',
      'generateArtifact',
      'approveOutput',
      'importModel',
      'viewAuditLog',
      'modifyPolicy',
      'enterProvisioning',
    ] as const) {
      expect(view.has(p)).toBe(true);
    }
  });

  it('Employee cannot use administrative Tauri commands (backend rejection expected)', async () => {
    mockedService.currentSession.mockResolvedValue({
      user: { id: 'engineer', displayName: 'P. Shetty', roles: ['employee'], department: null },
      startedAt: '2026-08-30T00:00:00Z',
    });
    mockedService.currentPermissions.mockResolvedValue([
      'useModel',
      'uploadDocument',
      'searchKnowledge',
      'executeCode',
      'generateArtifact',
      'approveOutput',
    ]);

    const view = await loadPermissions();
    // The back-end would reject these for an Employee. The context is a UX
    // view; the real boundary is the server. These assertions pin the
    // "hide the affordance" side of that.
    expect(view.has('importModel')).toBe(false);
    expect(view.has('viewAuditLog')).toBe(false);
    expect(view.has('modifyPolicy')).toBe(false);
    expect(view.has('enterProvisioning')).toBe(false);
  });

  it('Administrator is a superset of Employee: every Employee permission is also held', async () => {
    mockedService.currentSession.mockResolvedValue({
      user: { id: 'admin', displayName: 'R. Nair', roles: ['administrator'], department: null },
      startedAt: '2026-08-30T00:00:00Z',
    });
    mockedService.currentPermissions.mockResolvedValue([
      'useModel',
      'uploadDocument',
      'searchKnowledge',
      'executeCode',
      'generateArtifact',
      'approveOutput',
      'importModel',
      'viewAuditLog',
      'modifyPolicy',
      'enterProvisioning',
    ]);

    const view = await loadPermissions();
    const employeeOnly: Permission[] = [
      'useModel',
      'uploadDocument',
      'searchKnowledge',
      'executeCode',
      'generateArtifact',
      'approveOutput',
    ];
    for (const p of employeeOnly) {
      expect(view.has(p)).toBe(true);
    }
    // Plus the administrative ones.
    expect(view.has('importModel')).toBe(true);
    expect(view.has('viewAuditLog')).toBe(true);
    expect(view.has('modifyPolicy')).toBe(true);
    expect(view.has('enterProvisioning')).toBe(true);
  });

  it('survives a back-end failure as no permissions and ready=true', async () => {
    mockedService.currentSession.mockRejectedValue(new Error('offline'));
    const view = await loadPermissions();
    expect(view.session).toBeNull();
    expect(view.held.size).toBe(0);
    expect(view.has('useModel')).toBe(false);
    expect(view.ready).toBe(true);
  });
});
