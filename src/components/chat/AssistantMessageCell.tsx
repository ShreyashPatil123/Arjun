import React, { useState, useMemo } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  CircleDashed,
  ChevronDown,
  CircleSlash,
  Copy,
  Eye,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Loader2,
  RotateCcw,
  ShieldCheck,
  X,
} from 'lucide-react';
import {
  agentService,
  messageStatus,
  MESSAGE_STATUS_LABELS,
  type MessageStatusKind,
  type ArtifactPreview,
  type ArtifactReport,
  type ChatMessage,
  type RunSummary,
} from '../../services/agent.service';
import { formatDuration, formatTokens } from './format';
import { collapseForDisplay } from '../../contexts/ConversationContext';
import { iconForTool, labelForTool } from '../../services/toolNames';
import { ChatOrb } from './ChatOrb';
import { ThinkingTree, type ThinkingNode } from './ThinkingTree';
import { RunProgressPanel } from './RunProgressPanel';
import type { ProgressStep } from './runProgress';
import { useTokenMetrics, type TokenMetrics } from './useTokenMetrics';
import { Markdown } from './Markdown';
import { ReasoningStream } from './ReasoningStream';
import type { LiveReasoning } from '../../contexts/ConversationContext';
import styles from './ChatSurface.module.css';

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function parseThinking(content: string): { reasoning: string; answer: string; nodes: ThinkingNode[] } {
  let reasoning = '';
  let answer = content;

  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
  if (thinkMatch) {
    reasoning = thinkMatch[1].trim();
    answer = content.replace(thinkMatch[0], '').trim();
  }

  const nodes: ThinkingNode[] = [];
  if (reasoning) {
    const lines = reasoning.split('\n').filter(l => l.trim());
    lines.forEach((line, idx) => {
      const trimmed = line.trim().replace(/^[-·]\s*/, '');
      nodes.push({
        id: `r-${idx}`,
        label: trimmed,
        status: 'done',
        icon: 'none',
      });
    });
  }

  return { reasoning, answer, nodes };
}

/**
 * The full reading, for the hover the pill cannot fit.
 *
 * The pill shows output tokens and the rate because those are what change
 * while an answer is being written. The prompt size and whether the numbers
 * were reported or estimated matter too, but only to someone who has stopped
 * to look — so they live here rather than widening the line.
 */
function tokenTitle(metrics: TokenMetrics): string {
  const parts: string[] = [];
  if (metrics.tokensIn > 0) parts.push(`${metrics.tokensIn} tokens in`);
  parts.push(
    metrics.approx
      ? `about ${metrics.tokensOut} tokens out, estimated from the text — this model reported no usage`
      : `${metrics.tokensOut} tokens out`,
  );
  if (metrics.speed > 0) parts.push(`${metrics.speed} tokens/second`);
  return parts.join(' · ');
}

function StatusPill({
  state,
  elapsedText,
  runningTools,
  metrics,
}: {
  state: MessageStatusKind;
  elapsedText: string | null;
  runningTools: number;
  metrics: TokenMetrics;
}) {
  const labels: Record<MessageStatusKind, string> = {
    ...MESSAGE_STATUS_LABELS,
    // The only label that depends on something outside the status itself.
    usingTool: runningTools > 1 ? `Using ${runningTools} tools…` : 'Using a tool…',
  };

  return (
    <span className={styles.statusPill} data-state={state}>
      {state === 'thinking' || state === 'usingTool' || state === 'composing' ? (
        <Loader2 size={11} className={styles.spin} />
      ) : state === 'verified' ? (
        // The tick is reserved for the one state that earned it: the verifier
        // ran and every claim resolved.
        <CheckCircle2 size={11} />
      ) : state === 'failed' ? (
        <X size={11} />
      ) : (
        // Finished, but not certified: needs review, unverified, stopped, or
        // completed with nothing to check. None of those is a failure and none
        // of them is a pass.
        <CircleDashed size={11} />
      )}
      <span>{labels[state]}</span>
      {elapsedText && <span className={styles.statusPillSub}>· {elapsedText}</span>}
      {(state === 'composing' ||
        state === 'verified' ||
        state === 'needsReview' ||
        state === 'unverified') &&
        metrics.tokensOut > 0 && (
        <span className={styles.statusPillSub} title={tokenTitle(metrics)}>
          · {metrics.approx ? '~' : ''}
          {formatTokens(metrics.tokensOut)} tok
          {metrics.speed > 0 && ` · ${metrics.speed} tok/s`}
        </span>
      )}
    </span>
  );
}

interface AssistantMessageCellProps {
  message: ChatMessage;
  liveContent?: string;
  isLive?: boolean;
  activity?: {
    id: string;
    tool: string;
    status: 'running' | 'done' | 'failed' | 'refused' | 'replayed' | 'unknown';
    startedAt?: number;
    endedAt?: number;
    inputSummary?: string;
    outputSummary?: string;
    artifactPath?: string;
    errorMessage?: string;
  }[];
  runSummary?: RunSummary | null;
  /**
   * What this turn has been doing, newest last.
   *
   * Keyed to this message by the reducer, never matched by position in the
   * log: a list found by index is how one turn's progress ends up under
   * another turn's answer.
   */
  progress?: ProgressStep[];
  /**
   * The reasoning this turn has produced so far.
   *
   * Live only, and absent on every finished message: it is held in the
   * reducer for the life of the run and never persisted, so a reopened
   * conversation shows the answer with no thinking behind it.
   */
  reasoning?: LiveReasoning;
  /** Only the newest assistant cell draws the orb. */
  showAvatar?: boolean;
  onOpenInspector?: (runId: string) => void;
  onRetry?: () => void;
  composerDisabled?: boolean;
}

export function AssistantMessageCell({
  message,
  liveContent,
  isLive,
  activity,
  runSummary,
  progress,
  // Renamed on the way in. `reasoning` is already spoken for in this
  // component: `parseThinking` returns the inline <think> block under that
  // name, and the two are different things — one was streamed on its own
  // channel, the other was dug out of the answer after the fact.
  reasoning: liveReasoning,
  showAvatar,
  onOpenInspector,
  onRetry,
}: AssistantMessageCellProps) {
  // What was stored: the model's exact words, in the order it produced them.
  //
  // Repetition is collapsed for *display* only, and only when the answer has no
  // fenced code in it. It used to be collapsed inside the streaming reducer,
  // which meant the edited text was what got persisted, sent as `finalContent`,
  // resolved against by the verifier and written into the audit record — a
  // display convenience editing the evidence. `stored` is what everything else
  // in the product sees; `content` is what this cell draws.
  const stored = isLive ? (liveContent ?? message.content) : message.content;
  const display = useMemo(() => collapseForDisplay(stored), [stored]);
  const content = display.text;
  const isStreaming = isLive === true || message.status === 'streaming';
  const isFailed = message.status === 'failed';
  const isDone = !isStreaming && !isFailed;

  const elapsedMs = message.elapsedMs ?? null;
  const elapsedText = elapsedMs !== null ? formatDuration(elapsedMs) : null;
  const modelName = message.modelName ?? runSummary?.routing.modelName ?? 'model';
  const modelRole = message.modelRole ?? runSummary?.routing.role ?? null;
  const usedFallback = message.usedFallback ?? runSummary?.routing.usedFallback ?? false;

  const runningTools = activity?.filter(a => a.status === 'running').length ?? 0;
  const toolsTotal = activity?.length ?? 0;

  // What this turn actually is, from what was actually recorded.
  //
  // This used to read `isFailed ? 'failed' : isDone ? 'verified' : ...`, so
  // "not streaming and not failed" was rendered as **Verified**, with a green
  // tick, for every turn — including ones the verifier never looked at, ones
  // it found blocking problems in, and ones a person stopped part way through.
  // The strongest claim the product can make was the one it made by default.
  const status = messageStatus({
    isStreaming,
    contentLength: content.length,
    runningTools,
    // Persisted per message, so a cell rendered from disk long after the run's
    // events have gone still knows how it ended and what checked it.
    outcome: message.outcome ?? (isFailed ? 'failed' : null),
    verification:
      message.verification ??
      (runSummary?.verification
        ? runSummary.verification.standing.standing === 'ready'
          ? 'ready'
          : 'needsReview'
        : null),
  });

  const [copied, setCopied] = useState(false);

  const metrics = useTokenMetrics(
    isStreaming,
    content.length,
    message.tokensIn,
    message.tokensOut,
    elapsedMs,
  );

  // FIX 3: Parse thinking / reasoning.
  const { reasoning, answer, nodes } = useMemo(
    () => parseThinking(content),
    [content],
  );
  const displayContent = reasoning ? answer : content;

  // The timeline for this turn: the model's own steps first, then the
  // tool calls the run actually made. Two `group` keys means two cards,
  // which is how thinking and doing read as separate passes of work.
  const timelineNodes: ThinkingNode[] = useMemo(() => {
    const out: ThinkingNode[] = nodes.map(node => ({
      ...node,
      group: 'reasoning',
      // Row labels are ellipsised to one line, so anything long enough
      // to be cut keeps its full text behind the chevron.
      detail: node.label.length > 90 ? node.label : undefined,
    }));
    for (const item of activity ?? []) out.push(toolNode(item));
    return out;
  }, [nodes, activity]);

  return (
    <div className={styles.assistantRow}>
      {/* Left column: the orb on the newest cell only, and an empty
        * slot of the same width on the rest so every message in the log
        * stays on one left edge. */}
      <div className={styles.assistantAvatar}>
        {showAvatar && <ChatOrb active={isStreaming} size={36} />}
      </div>

      {/* RIGHT: Meta + Content */}
      <div className={styles.assistantCol}>
        <div className={styles.assistantMeta}>
          <StatusPill
            state={status}
            elapsedText={elapsedText}
            runningTools={runningTools}
            metrics={metrics}
          />
          <span className={styles.assistantMetaSep}>·</span>
          <span className={styles.assistantModel}>
            {modelName}
            {modelRole && <span className={styles.assistantModelRole}> · {modelRole}</span>}
            {usedFallback && (
              <span className={styles.assistantFallback}>fallback</span>
            )}
          </span>
        </div>

        <div className={styles.assistantBody}>
          {isFailed ? (
            <p className={styles.errorLine}>
              <AlertTriangle size={13} />
              <span>{message.error ?? 'The run did not finish cleanly.'}</span>
            </p>
          ) : (
            <>
              {/* What the turn is doing, above the answer. Rendered before
                * the tool timeline because it covers the earlier interval:
                * the reading, routing and loading that happen before the
                * model is asked anything. */}
              {progress && progress.length > 0 && (
                <RunProgressPanel
                  steps={progress}
                  isLive={isStreaming}
                  hasAnswer={displayContent.length > 0}
                />
              )}

              {/* The model thinking, while it thinks. Above the timeline
                * and the answer because it covers the interval between
                * them: the request has gone, the answer has not started,
                * and this is the only thing happening. It closes itself
                * when the answer begins. */}
              <ReasoningStream
                reasoning={liveReasoning}
                isLive={isStreaming}
                hasAnswer={displayContent.length > 0}
              />

              {/* The record of the turn. It stays after the run ends:
                * this is what happened, not a progress spinner. */}
              {timelineNodes.length > 0 && (
                <ThinkingTree
                  nodes={timelineNodes}
                  isLive={isStreaming}
                  summary={elapsedText}
                />
              )}

              {/* FIX 2 & 6: Token-by-token streaming with markdown rendering. */}
              <div
                className={styles.assistantText}
                aria-live={isStreaming ? 'polite' : undefined}
                aria-busy={isStreaming || undefined}
              >
                {displayContent ? (
                  <Markdown content={displayContent} />
                ) : isStreaming ? (
                  <span className={styles.assistantPlaceholder} aria-hidden="true" />
                ) : null}
                {isStreaming && displayContent && (
                  <span className={styles.caret} aria-hidden="true" />
                )}
              </div>

              {/* Per-message actions sit after the text in the DOM so a
                * screen reader reaches the answer before the controls. */}
              {isDone && displayContent.length > 0 && (
                <div className={styles.messageActions}>
                  <button
                    type="button"
                    className={styles.messageAction}
                    onClick={() => {
                      void navigator.clipboard
                        .writeText(displayContent)
                        .then(() => {
                          setCopied(true);
                          window.setTimeout(() => setCopied(false), 1500);
                        })
                        .catch(() => setCopied(false));
                    }}
                    aria-label={copied ? 'Copied to clipboard' : 'Copy this answer'}
                  >
                    {copied ? <Check size={12} /> : <Copy size={12} />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
              )}
            </>
          )}

          {isFailed && onRetry && (
            <button className={styles.retryBtn} onClick={onRetry} type="button">
              <RotateCcw size={12} /> Retry
            </button>
          )}

          {runSummary && runSummary.artifacts.length > 0 && (
            <ArtifactList runId={runSummary.runId} artifacts={runSummary.artifacts} />
          )}

          {message.runId && onOpenInspector && (
            <div className={styles.assistantFooter}>
              <button
                className={styles.detailsBtn}
                onClick={() => onOpenInspector(message.runId!)}
                type="button"
              >
                View details ?
              </button>
              <span className={styles.assistantFooterMeta}>
                {toolsTotal > 0 && <>{toolsTotal} tool{toolsTotal === 1 ? '' : 's'} · </>}
                {runSummary && <>{runSummary.plan.steps.length} step{runSummary.plan.steps.length === 1 ? '' : 's'} · </>}
                <ShieldCheck size={10} />
                <span>verified</span>
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface ActivityEntry {
  id: string;
  tool: string;
  status: 'running' | 'done' | 'failed' | 'refused' | 'replayed' | 'unknown';
  startedAt?: number;
  endedAt?: number;
  inputSummary?: string;
  outputSummary?: string;
  artifactPath?: string;
  errorMessage?: string;
}

const STATUS_LABEL: Record<ActivityEntry['status'], string> = {
  running: 'running',
  done: 'done',
  failed: 'failed',
  refused: 'not permitted',
  replayed: 'already done',
  unknown: 'interrupted',
};

// The labels and the icon rule both live in `services/toolNames.ts`. They used
// to be a second copy of the table in `useRun.ts`, keyed on the pre-namespace
// spelling, so a live event carrying the current name fell through to the raw
// string. See that module.
function toolLabel(tool: string) {
  return labelForTool(tool);
}

function toolIcon(tool: string): ThinkingNode['icon'] {
  return iconForTool(tool);
}

/**
 * What the chevron reveals for a tool row: what the tool was asked,
 * what came back, what it produced, why it failed. Anything the run
 * did not record is simply left out rather than shown as an empty
 * field.
 */
function toolDetail(item: ActivityEntry): string | undefined {
  const parts: string[] = [];
  if (item.inputSummary) parts.push(`Asked: ${item.inputSummary}`);
  if (item.outputSummary) parts.push(`Returned: ${item.outputSummary}`);
  if (item.artifactPath) parts.push(`Produced: ${item.artifactPath}`);
  if (item.errorMessage) parts.push(`Failed because: ${item.errorMessage}`);
  if (parts.length === 0 && item.status !== 'done' && item.status !== 'running') {
    parts.push(STATUS_LABEL[item.status]);
  }
  return parts.length > 0 ? parts.join('\n') : undefined;
}

/** One backend activity record as a timeline row. */
function toolNode(item: ActivityEntry): ThinkingNode {
  const duration =
    item.endedAt && item.startedAt
      ? formatDuration(Math.max(0, item.endedAt - item.startedAt))
      : undefined;
  return {
    id: item.id,
    label: toolLabel(item.tool),
    group: 'tools',
    icon: toolIcon(item.tool),
    status:
      item.status === 'running'
        ? 'running'
        : item.status === 'done' || item.status === 'replayed'
          ? 'done'
          : 'failed',
    meta: item.status === 'running' ? 'in progress' : duration,
    detail: toolDetail(item),
  };
}

const ARTIFACT_ICONS: Record<ArtifactReport['kind'], typeof FileText> = {
  document: FileText,
  workbook: FileSpreadsheet,
  text: FileText,
};

function ArtifactList({ runId, artifacts }: { runId: string; artifacts: ArtifactReport[] }) {
  return (
    <ul className={styles.artifactList}>
      {artifacts.map(artifact => (
        <ArtifactRow key={artifact.path} runId={runId} artifact={artifact} />
      ))}
    </ul>
  );
}

function ArtifactRow({ runId, artifact }: { runId: string; artifact: ArtifactReport }) {
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<ArtifactPreview | 'loading' | 'error' | undefined>(undefined);
  const [problem, setProblem] = useState<string | null>(null);
  const Icon = ARTIFACT_ICONS[artifact.kind];

  const reveal = async () => {
    try {
      await agentService.revealArtifact(runId, artifact.name);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    }
  };

  const togglePreview = async () => {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (preview && preview !== 'error') return;
    setPreview('loading');
    try {
      const p = await agentService.previewArtifact(runId, artifact.name);
      setPreview(p);
    } catch {
      setPreview('error');
    }
  };

  return (
    <li className={styles.artifactRow}>
      <div className={styles.artifactRowMain}>
        <Icon size={13} className={styles.artifactIcon} />
        <button type="button" className={styles.artifactNameBtn} onClick={togglePreview} title={open ? 'Hide preview' : 'Preview'}>
          <span className={styles.artifactName}>{artifact.name}</span>
        </button>
        <span className={styles.artifactSize}>{size(artifact.bytes)}</span>
        <span className={artifact.sound ? styles.tagSound : styles.tagUnsound}>
          {artifact.sound ? 'opens and checks out' : 'did not pass its check'}
        </span>
        <div className={styles.artifactActions}>
          <button type="button" className={styles.iconBtn} onClick={togglePreview} aria-label={open ? 'Hide preview' : 'Preview'} title={open ? 'Hide preview' : 'Preview'}>
            {open ? <ChevronDown size={12} /> : <Eye size={12} />}
          </button>
          <button type="button" className={styles.iconBtn} onClick={reveal} aria-label="Show in file manager" title="Show in file manager">
            <FolderOpen size={12} />
          </button>
        </div>
      </div>
      {open && <ArtifactPreviewPane preview={preview} name={artifact.name} />}
      {problem && (
        <p className={styles.errorLine} role="alert">
          <AlertTriangle size={12} />
          <span>{problem}</span>
        </p>
      )}
    </li>
  );
}

function ArtifactPreviewPane({ preview, name }: { preview: ArtifactPreview | 'loading' | 'error' | undefined; name: string }) {
  if (preview === undefined || preview === 'loading') {
    return (
      <div className={styles.previewPane} aria-busy="true">
        <Loader2 size={12} className={styles.spin} />
        <span>Reading {name}·</span>
      </div>
    );
  }
  if (preview === 'error') {
    return (
      <div className={styles.previewPane}>
        <AlertTriangle size={12} />
        <span>Could not load a preview of {name}.</span>
      </div>
    );
  }
  if (preview.kind === 'unsupported') {
    return (
      <div className={styles.previewPane}>
        <CircleSlash size={12} />
        <span>Preview not available for this format ({preview.mime || 'unknown'}). Use the folder button to open it in the file manager.</span>
      </div>
    );
  }
  if (preview.kind === 'image') {
    return (
      <div className={styles.previewPane}>
        <img className={styles.previewImage} src={preview.dataUrl} alt={`Preview of ${name}`} />
        {preview.truncated && <p className={styles.previewNote}>Preview is truncated to fit.</p>}
      </div>
    );
  }
  const mono = preview.kind === 'docxBody' || preview.kind === 'xlsxFirstSheet' || preview.kind === 'text';
  return (
    <div className={styles.previewPane}>
      <pre className={mono ? styles.previewPre : styles.previewMarkdown} data-truncated={preview.truncated || undefined}>
        {preview.content}
      </pre>
      {preview.truncated && <p className={styles.previewNote}>Preview is truncated. Use the folder button for the full file.</p>}
    </div>
  );
}