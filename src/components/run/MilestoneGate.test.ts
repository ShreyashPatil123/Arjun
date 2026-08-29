// filepath: src/components/run/MilestoneGate.test.ts
import { describe, expect, it, vi } from 'vitest';
import { agentService } from '../../services/agent.service';
import type { MilestoneGate as Gate } from './recovery';

/**
 * Service contract for the milestone gate.
 *
 * The component itself uses Testing Library and a DOM, which the
 * vitest config deliberately does not pull in. The service is the
 * other half of the contract — what the UI hands the backend when a
 * person decides on a checkpoint — and that is the part that is
 * worth testing here.
 */
const gate: Gate = {
  checkpointId: 'mtn-survey',
  ordinal: 2,
  summary: 'Surveyed the SOPs and the inspection reports.',
};

describe('milestone acknowledgement service contract', () => {
  it('the run id, checkpoint id, and decision all reach the backend', async () => {
    const spy = vi
      .spyOn(agentService, 'acknowledgeMilestone')
      .mockResolvedValue({
        checkpointId: 'mtn-survey',
        ordinal: 2,
        decision: 'approved',
        acknowledgedBy: 'priya',
        at: '2026-08-27T10:00:00+00:00',
      });

    await agentService.acknowledgeMilestone('run-7', gate.checkpointId, 'approved');
    expect(spy).toHaveBeenCalledWith('run-7', 'mtn-survey', 'approved');

    await agentService.acknowledgeMilestone('run-7', gate.checkpointId, 'rejected');
    expect(spy).toHaveBeenCalledWith('run-7', 'mtn-survey', 'rejected');

    spy.mockRestore();
  });

  it('a rejection is a deliberate end, not a failure: the run stops with the work preserved', async () => {
    const spy = vi
      .spyOn(agentService, 'acknowledgeMilestone')
      .mockResolvedValue({
        checkpointId: 'mtn-survey',
        ordinal: 2,
        decision: 'rejected',
        acknowledgedBy: 'priya',
        at: '2026-08-27T10:00:00+00:00',
      });

    const result = await agentService.acknowledgeMilestone(
      'run-7',
      gate.checkpointId,
      'rejected',
    );
    expect(result.decision).toBe('rejected');
    expect(result.checkpointId).toBe(gate.checkpointId);
    spy.mockRestore();
  });
});
