import React, { useCallback, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Mic, ArrowUp, ShieldCheck, ShieldAlert } from 'lucide-react';
import { sovereigntyService } from '../services/sovereignty.service';
import { RoutingPreview } from '../components/routing/RoutingPreview';
import { RunView } from '../components/run/RunView';
import { isBusy, useRun } from '../components/run/useRun';
import { registryService, type PreparedModel } from '../services/registry.service';
import styles from './Workbench.module.css';

/** Grows the composer with its content up to a cap, after which it scrolls.
 *  Height is reset before measuring or the box can only ever grow. */
const MAX_COMPOSER_HEIGHT = 220;

export const Workbench = () => {
  const [prompt, setPrompt] = useState('');
  const [attachments, setAttachments] = useState<string[]>([]);
  // Set when the invariant refused an attachment, so the reason is shown rather
  // than the control simply doing nothing.
  const [refusal, setRefusal] = useState<string | null>(null);
  /** Set while the routed model is being loaded — a swap takes seconds. */
  const [preparing, setPreparing] = useState(false);
  const [prepared, setPrepared] = useState<PreparedModel | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { state: run, start, abort, reset } = useRun();

  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_COMPOSER_HEIGHT)}px`;
  }, []);

  const busy = preparing || isBusy(run.phase);
  const canSubmit = prompt.trim().length > 0 || attachments.length > 0;

  const submit = useCallback(async () => {
    if (!canSubmit || busy) return;
    const asked = prompt.trim();

    // Routing and loading happen first, so by the time the run starts the right
    // model is already resident. This is the automatic selection PS 26117 asks
    // to be demonstrated: no human step between asking and the correct model
    // being ready.
    setPreparing(true);
    setRefusal(null);
    try {
      setPrepared(await registryService.prepareModelFor(asked));
    } catch (e) {
      // Nothing to run against, so the run is not started. Saying why and
      // stopping is better than starting a task that will fail on its first
      // turn for a reason this screen already knows.
      setPrepared(null);
      setRefusal(e instanceof Error ? e.message : String(e));
      setPreparing(false);
      return;
    }
    setPreparing(false);

    // The backend routes again, and its decision is the one that counts —
    // `prepareModelFor` above is what makes the model resident, not what
    // chooses it. Handing a model down from here would make the automatic
    // selection optional, which is the opposite of the point.
    await start(asked);
  }, [busy, canSubmit, prompt, start]);

  /** Clears the run and returns to the composer, ready for the next task. */
  const newTask = useCallback(() => {
    reset();
    setPrompt('');
    setAttachments([]);
    setPrepared(null);
    setRefusal(null);
  }, [reset]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter breaks the line. Matches what people already
    // expect from the assistants this is meant to replace.
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submit();
    }
  };

  const onFilesPicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(e.target.files ?? []).map(f => f.name);
    // Clear the input so picking the same file twice still fires a change.
    e.target.value = '';
    if (!picked.length) return;

    // An attached document is confidential material entering the process, so it
    // is refused whenever the network is reachable. Checked before the names are
    // even held in state.
    try {
      await sovereigntyService.assertConfidentialAllowed('attaching a document');
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
      return;
    }

    setRefusal(null);
    setAttachments(prev => [...prev, ...picked]);
  };

  // Once a task is under way the composer gives up the screen to it. The two
  // are deliberately not shown together: a run takes minutes and produces a
  // page of evidence, and a box inviting a second task above it would suggest
  // the first can be left unread.
  if (run.phase !== 'idle') {
    return (
      <div className={styles.runPage}>
        <RunView state={run} onAbort={() => void abort()} onNewTask={newTask} />
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.centre}>
        <h1 className={styles.headline}>What can I do for you?</h1>

        <div className={styles.composer}>
          {attachments.length > 0 && (
            <ul className={styles.attachments}>
              {attachments.map((name, i) => (
                <li key={`${name}-${i}`} className={styles.chip}>
                  <span className={styles.chipName}>{name}</span>
                  <button
                    className={styles.chipRemove}
                    aria-label={`Remove ${name}`}
                    onClick={() => setAttachments(prev => prev.filter((_, j) => j !== i))}
                  >
                    &times;
                  </button>
                </li>
              ))}
            </ul>
          )}

          <textarea
            ref={textareaRef}
            className={styles.input}
            placeholder="Ask ARJUN anything — nothing leaves this machine"
            value={prompt}
            rows={1}
            onChange={e => { setPrompt(e.target.value); resize(); }}
            onKeyDown={onKeyDown}
            aria-label="Task prompt"
          />

          <div className={styles.controls}>
            <button
              className={styles.iconBtn}
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach a document, drawing or photograph"
            >
              <Plus size={19} />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              hidden
              onChange={onFilesPicked}
            />

            <div className={styles.controlsRight}>
              <button className={styles.iconBtn} aria-label="Dictate">
                <Mic size={18} />
              </button>
              <button
                className={styles.send}
                onClick={() => void submit()}
                disabled={!canSubmit || busy}
                aria-label="Start task"
              >
                <ArrowUp size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Which model will take this, decided and shown before anything runs. */}
        <RoutingPreview prompt={prompt} />

        {prepared && (
          <p className={styles.prepared} role="status">
            <strong>{prepared.activation.modelName}</strong>{' '}
            {prepared.activation.alreadyResident
              ? 'was already loaded'
              : prepared.activation.evicted
                ? `loaded, releasing ${prepared.activation.evicted}`
                : 'loaded'}
            {' · '}
            {prepared.routing.reasons[0]}
          </p>
        )}

        {refusal && (
          <p className={styles.refusal} role="alert">
            <ShieldAlert size={15} />
            <span>{refusal}</span>
          </p>
        )}

        {/* PS step 16 asks for a visible local-only indicator. It deliberately
          * does not claim the task *is* secure — it reports the mode, and links
          * to the monitor where the claim can actually be checked. */}
        <Link className={styles.localBadge} to="/audit">
          <ShieldCheck size={14} />
          <span>Work mode &middot; no external calls</span>
        </Link>
      </div>
    </div>
  );
};
