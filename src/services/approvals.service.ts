import { getBackendService } from './api';

/**
 * What a person is being asked to allow.
 *
 * Every field is captured when the request is raised, not derived afterwards.
 * A summary written later — from the tool call, or from the model's account of
 * it — is a different document from the one somebody actually read before
 * signing, and only the first is evidence.
 */
export interface ApprovalRequest {
  id: string;
  taskId: string;
  /** The tool that wants to run. */
  tool: string;
  /** What it would act on — a path, a document, a recipient. */
  target: string;
  /** The arguments, as the approver reads them. */
  arguments: string[];
  /** Passages and calculations the proposed action rests on, cited. */
  evidence: string[];
  expectedOutput: string;
  /** What it would change, and what could not be undone. */
  consequences: string;
  requestedBy: string;
  /** ISO-8601, from the backend clock. */
  requestedAt: string;
}

export type Decision =
  | { decision: 'approved'; by: string; at: string }
  | { decision: 'rejected'; by: string; at: string; because: string };

export interface ApprovalItem {
  request: ApprovalRequest;
  /** Absent while the request is still waiting. */
  decision: Decision | null;
}

export const approvalsService = {
  /** Everything raised this session, newest first, settled ones included. */
  list(): Promise<ApprovalItem[]> {
    return getBackendService().invoke<ApprovalItem[]>('list_approvals');
  },

  /**
   * Approves or rejects one request.
   *
   * Rejects with the refusal text when the signed-in user is not a reviewer,
   * when a rejection carries no reason, or when the request was already
   * settled — a decision is final by design.
   */
  decide(id: string, approve: boolean, because?: string): Promise<Decision> {
    return getBackendService().invoke<Decision>('decide_approval', { id, approve, because });
  },
};
