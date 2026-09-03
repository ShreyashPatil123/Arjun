/**
 * What the chat opens with, and where a scripted turn goes.
 *
 * Both decisions used to live inside a mount-only `useEffect`, reading values
 * that effect had closed over. A mount-only closure holds the state as it was
 * at mount, so both read `conversation === null` for the life of the session.
 * Pulled out as functions over their inputs, they can be driven here without a
 * DOM — and, more to the point, they cannot read a render at all.
 */

import { describe, expect, it } from 'vitest';
import { restoreOrCreate, runExclusive, targetForTrigger } from './ConversationContext';
import type { Conversation } from '../services/agent.service';

/**
 * The composer's send lock.
 *
 * ## The defect
 *
 * The lock was taken, and the `try` that released it did not begin until
 * several awaits later — after creating a conversation, after reserving the
 * turn, after refreshing the list, after registering the reducer. Any of those
 * throwing left the lock set with nothing to clear it, and the composer was
 * dead for the rest of the session: every later send returned early and did
 * nothing at all. No error, no spinner, no way for the person to tell that
 * their typing was going nowhere.
 *
 * The point of every test below is the *second* send. A lock that releases but
 * leaves the composer unusable would pass a test that only checked the flag.
 */
describe('runExclusive: the send lock always clears', () => {
  /** The pre-runtime awaits, in the order `sendTo` performs them. */
  const PRE_RUNTIME_STEPS = [
    'creating the conversation',
    'reserving the turn',
    'refreshing the list',
    'registering the reducer',
    'starting the runtime',
  ];

  it('releases the lock when the work succeeds', async () => {
    const lock = { current: false };
    const result = await runExclusive(lock, async () => 'done');
    expect(result).toBe('done');
    expect(lock.current).toBe(false);
  });

  it('releases the lock when any pre-runtime step throws, and the next send succeeds', async () => {
    // Failure injection, one step at a time. Each is a real await in `sendTo`
    // that used to sit outside the `try`.
    for (const step of PRE_RUNTIME_STEPS) {
      const lock = { current: false };

      await expect(
        runExclusive(lock, async () => {
          throw new Error(`the backend failed while ${step}`);
        }),
      ).rejects.toThrow(step);

      expect(lock.current, `the lock was left set after failing while ${step}`).toBe(false);

      // The part that matters: the composer still works.
      const next = await runExclusive(lock, async () => 'the next turn ran');
      expect(next, `the composer was dead after failing while ${step}`).toBe(
        'the next turn ran',
      );
      expect(lock.current).toBe(false);
    }
  });

  it('refuses a second turn while one is genuinely in flight', async () => {
    // The lock still has to do its job. Releasing on failure must not become
    // never holding it.
    const lock = { current: false };
    let release: (() => void) | undefined;
    const inFlight = runExclusive(lock, () => new Promise<string>((resolve) => {
      release = () => resolve('first');
    }));

    expect(lock.current).toBe(true);
    expect(await runExclusive(lock, async () => 'second')).toBeUndefined();

    release?.();
    expect(await inFlight).toBe('first');
    expect(lock.current).toBe(false);
  });

  it('rethrows what the work threw, so the caller can still report it', async () => {
    // Releasing the lock must not swallow the reason. A turn that failed
    // silently is the failure mode one step along from a dead composer.
    const lock = { current: false };
    const thrown = new Error('the model server refused');
    await expect(runExclusive(lock, async () => Promise.reject(thrown))).rejects.toBe(thrown);
    expect(lock.current).toBe(false);
  });

  it('releases the lock even when the work throws synchronously', async () => {
    const lock = { current: false };
    await expect(
      runExclusive(lock, () => {
        throw new Error('threw before the first await');
      }),
    ).rejects.toThrow('threw before the first await');
    expect(lock.current).toBe(false);
  });
});

/** A conversation as the store returns it. */
function conversation(id: string, title = 'Thread'): Conversation {
  return {
    id,
    title,
    createdAt: '2026-08-27T10:00:00+00:00',
    lastActivityAt: '2026-08-27T10:00:00+00:00',
    messages: [],
    runs: [],
    compactions: 0,
  };
}

/**
 * Restoring the last conversation on mount.
 *
 * ## The defect
 *
 * The fallback was written `if (!conversation)`, against the state the
 * mount-only effect had captured — `null`, permanently. So it was *always*
 * true. A session that successfully restored its remembered thread went on to
 * create a second, empty one and made that the open conversation. The restored
 * thread was still on disk; it simply was not the one on screen, and the
 * person's history looked as though it had been lost.
 */
describe('restoreOrCreate: a restored conversation is the one that opens', () => {
  it('creates nothing when the remembered conversation is still there', async () => {
    const created: string[] = [];
    const result = await restoreOrCreate({
      remembered: 'conv-1',
      getConversation: async (id) => conversation(id, 'Yesterday'),
      createConversation: async (title) => {
        created.push(title);
        return conversation('conv-new', title);
      },
      forget: () => {
        throw new Error('a conversation that was found must not be forgotten');
      },
    });

    expect(result.conversation.id).toBe('conv-1');
    expect(result.created).toBe(false);
    expect(created, 'a second conversation was created over the restored one').toEqual([]);
  });

  it('creates exactly one when there is nothing remembered', async () => {
    const created: string[] = [];
    const result = await restoreOrCreate({
      remembered: null,
      getConversation: async () => {
        throw new Error('nothing to look up');
      },
      createConversation: async (title) => {
        created.push(title);
        return conversation('conv-new', title);
      },
      forget: () => {},
    });

    expect(created).toEqual(['New conversation']);
    expect(result.conversation.id).toBe('conv-new');
    expect(result.created).toBe(true);
  });

  it('forgets a remembered id that names nothing, and creates one', async () => {
    // Deleted, or belonging to another user. Not an error, and not a reason to
    // keep asking for it on every start.
    let forgotten = false;
    const result = await restoreOrCreate({
      remembered: 'conv-gone',
      getConversation: async () => null,
      createConversation: async (title) => conversation('conv-new', title),
      forget: () => {
        forgotten = true;
      },
    });

    expect(forgotten).toBe(true);
    expect(result.created).toBe(true);
    expect(result.conversation.id).toBe('conv-new');
  });

  it('treats a lookup that throws as a conversation that is gone', async () => {
    const result = await restoreOrCreate({
      remembered: 'conv-1',
      getConversation: async () => {
        throw new Error('the store is unavailable');
      },
      createConversation: async (title) => conversation('conv-new', title),
      forget: () => {},
    });
    expect(result.created).toBe(true);
  });
});

/**
 * Where a scripted `arjun:trigger-send` turn goes.
 *
 * ## The defect
 *
 * The handler created the titled conversation and then called `send`, whose
 * captured `conversation` was the mount-time `null`. `send` therefore created a
 * *second* conversation and put the turn in it. One demo click produced two
 * threads, and the titled one — the one the person was shown — stayed empty.
 */
describe('targetForTrigger: one demo event, one conversation', () => {
  it('creates exactly one conversation for a titled event and targets it', async () => {
    const created: string[] = [];
    const result = await targetForTrigger({
      title: 'P&ID review',
      current: null,
      createConversation: async (title) => {
        created.push(title);
        return conversation('conv-demo', title);
      },
    });

    expect(created, 'a titled demo event must create exactly one conversation').toEqual([
      'P&ID review',
    ]);
    expect(result.created).toBe(true);
    expect(result.conversation?.id).toBe('conv-demo');
    // The turn goes into the conversation that was just made — not into a
    // second one decided by a stale closure.
    expect(result.conversation?.title).toBe('P&ID review');
  });

  it('targets the titled conversation even when another one is already open', async () => {
    const result = await targetForTrigger({
      title: 'P&ID review',
      current: conversation('conv-open', 'Something else'),
      createConversation: async (title) => conversation('conv-demo', title),
    });
    expect(result.conversation?.id).toBe('conv-demo');
  });

  it('continues the open conversation for an untitled event', async () => {
    const created: string[] = [];
    const result = await targetForTrigger({
      current: conversation('conv-open'),
      createConversation: async (title) => {
        created.push(title);
        return conversation('conv-new', title);
      },
    });

    expect(created).toEqual([]);
    expect(result.created).toBe(false);
    expect(result.conversation?.id).toBe('conv-open');
  });

  it('reads the conversation it was handed, not one captured at mount', async () => {
    // The property that broke: the target is a value passed in at fire time.
    // A caller reading a ref supplies the live one; nothing here can see a
    // render that has since been replaced.
    const first = await targetForTrigger({
      current: conversation('conv-a'),
      createConversation: async (title) => conversation('conv-new', title),
    });
    const second = await targetForTrigger({
      current: conversation('conv-b'),
      createConversation: async (title) => conversation('conv-new', title),
    });
    expect(first.conversation?.id).toBe('conv-a');
    expect(second.conversation?.id).toBe('conv-b');
  });
});
