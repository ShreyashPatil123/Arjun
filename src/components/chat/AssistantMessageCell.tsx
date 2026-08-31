import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleSlash,
  Eye,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Wrench,
  X,
} from 'lucide-react';
import {
  agentService,
  type ArtifactPreview,
  type ArtifactReport,
  type ChatMessage,
  type RunSummary,
} from '../../services/agent.service';
import { formatDuration } from './format';
import styles from './ChatSurface.module.css';

/* ------------------------------------------------------------------ *
 * The compact assistant message cell.
 *
 * One cell per assistant turn. The redesign has three changes:
 *
 *  1. **Status pill** replaces the model badge. The model is now in a
 *     quieter meta footer (`model · role · elapsed`) so the eye goes
 *     to the content first and the metadata second.
 *  2. **Single-line tools**: the tool list is a single `verb + duration`
 *     line per tool, not a card. Tools stay in the cell only when they
 *     are short; long ones get a tiny expand chevron.
 *  3. **Inline artifacts**: the artifact card is rendered inline next
 *     to the cell, not as a sub-card. The `View details` link still
 *     opens the full inspector for the run.
 *
 * Status states:
 *  - Thinking…     before any tokens have arrived
 *  - Planning…     once the plan is in
 *  - Using tool…   while tools are running
 *  - Composing…    while text is streaming
 *  - Verifying…    during the verification pass
 *  - Verified      on success
 *  - failed        on error
 * ------------------------------------------------------------------ */

const ARTIFACT_ICONS: Record<ArtifactReport['kind'], typeof FileText> = {
  document: FileText,
  workbook: FileSpreadsheet,
  text: FileText,
};

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

interface AssistantMessageCellProps {
  message: ChatMessage;
  /** Live streaming content for this cell, when applicable. Falls back to
   *  the persisted `message.content` if absent (e.g. a remount). */
  liveContent?: string;
  /** True when this is the currently streaming cell. */
  isLive?: boolean;
  /** Activity for the run that produced this message. */
  activity?: {
    id: string;
    tool: string;
    status:
      | 'running'
      | 'done'
      | 'failed'
      | 'refused'
      | 'replayed'
      | 'unknown';
    startedAt?: number;
    endedAt?: number;
    inputSummary?: string;
    outputSummary?: string;
    artifactPath?: string;
    errorMessage?: string;
  }[];
  /** Run summary, for the per-run inspector. */
  runSummary?: RunSummary | null;
  /** Whether the inspector modal is currently open for this message. */
  onOpenInspector?: (runId: string) => void;
  /** Retry handler for failed messages. */
  onRetry?: () => void;
  /** True when the user is composing a follow-up; the retry button
   *  is hidden in that case. */
  composerDisabled?: boolean;
}

export function AssistantMessageCell({
  message,
  liveContent,
  isLive,
  activity,
  runSummary,
  onOpenInspector,
  onRetry,
}: AssistantMessageCellProps) {
  const content = isLive ? liveContent ?? message.content : message.content;
  const isStreaming = isLive || message.status === 'streaming';
  const isFailed = message.status === 'failed';

  const elapsedMs = message.elapsedMs ?? null;
  const elapsedText = elapsedMs !== null ? formatDuration(elapsedMs) : null;
  const modelName =
    message.modelName ?? runSummary?.routing.modelName ?? 'model';
  const modelRole =
    message.modelRole ?? runSummary?.routing.role ?? null;
  const usedFallback =
    message.usedFallback ?? runSummary?.routing.usedFallback ?? false;

  // Status derivation: which of the seven states are we in?
  const runningTools = activity?.filter(a => a.status === 'running').length ?? 0;
  const toolsTotal = activity?.length ?? 0;
  const status = isFailed
    ? 'failed'
    : isStreaming
      ? content.length > 0
        ? 'composing'
        : runningTools > 0
          ? 'usingTool'
          : 'thinking'
      : 'verified';

  return (
    <div className={styles.assistantRow}>
      {/* Meta footer — model identity and timing, quiet. */}
      <div className={styles.assistantMeta}>
        <StatusPill state={status} elapsedText={elapsedText} runningTools={runningTools} />
        <span className={styles.assistantMetaSep}>·</span>
        <span className={styles.assistantModel}>
          {modelName}
          {modelRole && (
            <span className={styles.assistantModelRole}> · {modelRole}</span>
          )}
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
          <p className={styles.assistantText}>
            {content || (isStreaming ? '…' : '')}
            {isStreaming && content && <span className={styles.caret}>▍</span>}
          </p>
        )}

        {isFailed && onRetry && (
          <button
            className={styles.retryBtn}
            onClick={onRetry}
            type="button"
          >
            <RotateCcw size={12} /> Retry
          </button>
        )}

        {activity && activity.length > 0 && (
          <ToolActivityList activity={activity} isLive={Boolean(isLive)} />
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
              View details →
            </button>
            <span className={styles.assistantFooterMeta}>
              {toolsTotal > 0 && (
                <>
                  {toolsTotal} tool{toolsTotal === 1 ? '' : 's'}
                  {' · '}
                </>
              )}
              {runSummary && (
                <>
                  {runSummary.plan.steps.length} step
                  {runSummary.plan.steps.length === 1 ? '' : 's'}
                  {' · '}
                </>
              )}
              <ShieldCheck size={10} />
              <span>verified</span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusPill({
  state,
  elapsedText,
  runningTools,
}: {
  state:
    | 'thinking'
    | 'usingTool'
    | 'composing'
    | 'verifying'
    | 'verified'
    | 'failed';
  elapsedText: string | null;
  runningTools: number;
}) {
  const labels: Record<typeof state, string> = {
    thinking: 'Thinking…',
    usingTool:
      runningTools > 1
        ? `Using ${runningTools} tools…`
        : 'Using a tool…',
    composing: 'Composing…',
    verifying: 'Verifying…',
    verified: 'Verified',
    failed: 'Failed',
  };
  return (
    <span className={styles.statusPill} data-state={state}>
      {state === 'thinking' || state === 'usingTool' || state === 'composing' || state === 'verifying' ? (
        <Loader2 size={11} className={styles.spin} />
      ) : state === 'verified' ? (
        <CheckCircle2 size={11} />
      ) : (
        <X size={11} />
      )}
      <span>{labels[state]}</span>
      {state === 'composing' && elapsedText && (
        <span className={styles.statusPillSub}>· {elapsedText}</span>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------ *
 * Tool activity list — single-line rows of tool calls the run made.
 *
 * Each tool reads as `verb · duration` on a single line. Failed tools
 * still show a single line plus a one-line reason. The expand chevron
 * only appears when there is something to expand (input/output/error).
 * ------------------------------------------------------------------ */

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

const STATUS_ICON: Record<
  ActivityEntry['status'],
  React.ComponentType<{ size?: number; className?: string }>
> = {
  running: Loader2,
  done: CheckCircle2,
  failed: AlertTriangle,
  refused: CircleSlash,
  replayed: CheckCircle2,
  unknown: AlertTriangle,
};

const STATUS_LABEL: Record<ActivityEntry['status'], string> = {
  running: 'running',
  done: 'done',
  failed: 'failed',
  refused: 'not permitted',
  replayed: 'already done',
  unknown: 'interrupted',
};

const TOOL_LABEL: Record<string, string> = {
  search_documents: 'Searching the documents',
  read_scoped_file: 'Reading a file',
  write_scoped_file: 'Writing a file',
  run_calculation: 'Calculating',
  create_docx: 'Producing a Word document',
  create_xlsx: 'Producing a workbook',
  execute_code: 'Running code',
  validate_artifact: 'Checking a produced file',
};

function toolLabel(tool: string) {
  return TOOL_LABEL[tool] ?? tool;
}

function ToolActivityList({
  activity,
  isLive,
}: {
  activity: ActivityEntry[];
  isLive: boolean;
}) {
  // Default to collapsed: a long list of tools is the one thing that
  // makes a chat unreadable, and the user can expand the ones they want.
  const [open, setOpen] = useState(false);
  const runningCount = activity.filter(a => a.status === 'running').length;
  const doneCount = activity.filter(a => a.status === 'done').length;
  const failedCount = activity.filter(a => a.status === 'failed').length;

  return (
    <div className={styles.toolBlock}>
      <button
        type="button"
        className={styles.toolSummary}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <Wrench size={11} className={styles.toolSummaryIcon} />
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <span className={styles.toolSummaryText}>
          {isLive && runningCount > 0
            ? `Using a tool… (${doneCount} done${failedCount > 0 ? `, ${failedCount} failed` : ''})`
            : `${activity.length} tool call${activity.length === 1 ? '' : 's'}`}
        </span>
      </button>
      {open && (
        <ul className={styles.toolList}>
          {activity.map(item => (
            <ToolRow key={item.id} item={item} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ToolRow({ item }: { item: ActivityEntry }) {
  const [expanded, setExpanded] = useState(item.status === 'failed');
  const Icon = STATUS_ICON[item.status];
  const duration =
    item.startedAt && item.endedAt
      ? formatDuration(Math.max(0, item.endedAt - item.startedAt))
      : item.status === 'running'
        ? 'in progress'
        : null;
  const expandable =
    Boolean(item.inputSummary) ||
    Boolean(item.outputSummary) ||
    Boolean(item.errorMessage) ||
    Boolean(item.artifactPath);

  return (
    <li className={styles.toolRow}>
      <div className={styles.toolRowMain}>
        <Icon size={11} className={styles[`statusIcon_${item.status}`]} />
        <span className={styles.toolName}>{toolLabel(item.tool)}</span>
        <span className={styles.toolStatus}>{STATUS_LABEL[item.status]}</span>
        {duration && <span className={styles.toolDuration}>{duration}</span>}
        {expandable && (
          <button
            type="button"
            className={styles.toolExpand}
            onClick={() => setExpanded(o => !o)}
            aria-expanded={expanded}
            aria-label={expanded ? 'Hide tool details' : 'Show tool details'}
          >
            {expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
          </button>
        )}
      </div>
      {expanded && expandable && (
        <div className={styles.toolDetail}>
          {item.inputSummary && (
            <div className={styles.toolDetailRow}>
              <span className={styles.toolDetailLabel}>Asked</span>
              <span className={styles.toolDetailValue}>{item.inputSummary}</span>
            </div>
          )}
          {item.outputSummary && (
            <div className={styles.toolDetailRow}>
              <span className={styles.toolDetailLabel}>Returned</span>
              <span className={styles.toolDetailValue}>{item.outputSummary}</span>
            </div>
          )}
          {item.artifactPath && (
            <div className={styles.toolDetailRow}>
              <span className={styles.toolDetailLabel}>Produced</span>
              <span className={`${styles.toolDetailValue} ${styles.toolPath}`}>
                {item.artifactPath}
              </span>
            </div>
          )}
          {item.errorMessage && (
            <div className={styles.toolDetailRow}>
              <span className={styles.toolDetailLabel}>Failed because</span>
              <span className={`${styles.toolDetailValue} ${styles.toolErrorLine}`}>
                {item.errorMessage}
              </span>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

/* ------------------------------------------------------------------ *
 * Artifact list — inline cards next to the cell. Each artifact has a
 * single line: icon · name · size · sound/unsound chip · actions.
 * Clicking the row opens an inline preview; the existing reveal button
 * is preserved for opening in the file manager.
 * ------------------------------------------------------------------ */

function ArtifactList({
  runId,
  artifacts,
}: {
  runId: string;
  artifacts: ArtifactReport[];
}) {
  return (
    <ul className={styles.artifactList}>
      {artifacts.map(artifact => (
        <ArtifactRow
          key={artifact.path}
          runId={runId}
          artifact={artifact}
        />
      ))}
    </ul>
  );
}

function ArtifactRow({
  runId,
  artifact,
}: {
  runId: string;
  artifact: ArtifactReport;
}) {
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<
    ArtifactPreview | 'loading' | 'error' | undefined
  >(undefined);
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
        <button
          type="button"
          className={styles.artifactNameBtn}
          onClick={togglePreview}
          title={open ? 'Hide preview' : 'Preview'}
        >
          <span className={styles.artifactName}>{artifact.name}</span>
        </button>
        <span className={styles.artifactSize}>{size(artifact.bytes)}</span>
        <span
          className={artifact.sound ? styles.tagSound : styles.tagUnsound}
        >
          {artifact.sound ? 'opens and checks out' : 'did not pass its check'}
        </span>
        <div className={styles.artifactActions}>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={togglePreview}
            aria-label={open ? 'Hide preview' : 'Preview'}
            title={open ? 'Hide preview' : 'Preview'}
          >
            {open ? <ChevronDown size={12} /> : <Eye size={12} />}
          </button>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={reveal}
            aria-label="Show in file manager"
            title="Show in file manager"
          >
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

function ArtifactPreviewPane({
  preview,
  name,
}: {
  preview: ArtifactPreview | 'loading' | 'error' | undefined;
  name: string;
}) {
  if (preview === undefined || preview === 'loading') {
    return (
      <div className={styles.previewPane} aria-busy="true">
        <Loader2 size={12} className={styles.spin} />
        <span>Reading {name}…</span>
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
        <span>
          Preview not available for this format ({preview.mime || 'unknown'}). Use
          the folder button to open it in the file manager.
        </span>
      </div>
    );
  }
  if (preview.kind === 'image') {
    return (
      <div className={styles.previewPane}>
        <img
          className={styles.previewImage}
          src={preview.dataUrl}
          alt={`Preview of ${name}`}
        />
        {preview.truncated && (
          <p className={styles.previewNote}>Preview is truncated to fit.</p>
        )}
      </div>
    );
  }
  const mono =
    preview.kind === 'docxBody' ||
    preview.kind === 'xlsxFirstSheet' ||
    preview.kind === 'text';
  return (
    <div className={styles.previewPane}>
      <pre
        className={mono ? styles.previewPre : styles.previewMarkdown}
        data-truncated={preview.truncated || undefined}
      >
        {preview.content}
      </pre>
      {preview.truncated && (
        <p className={styles.previewNote}>
          Preview is truncated. Use the folder button for the full file.
        </p>
      )}
    </div>
  );
}
