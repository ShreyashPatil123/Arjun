import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileSearch,
  Globe,
  Wrench,
} from 'lucide-react';

export interface ThinkingNode {
  id: string;
  label: string;
  icon?: 'search' | 'link' | 'tool' | 'none';
  meta?: string;
  status?: 'running' | 'done' | 'failed';
  children?: ThinkingNode[];
  href?: string;
  /** Splits the timeline into separate cards. Consecutive nodes that
   *  share a key sit in one card; a change of key opens a new one. */
  group?: string;
  /** Small monogram chip drawn after the label — the source a tool hit. */
  badge?: string;
  /** Extra text revealed when the row is expanded. */
  detail?: string;
}

interface ThinkingTreeProps {
  nodes: ThinkingNode[];
  isLive: boolean;
  /** Appended to the collapsed header, e.g. the run's elapsed time. */
  summary?: string | null;
}

/**
 * The reasoning tree: the running account of what the model is doing, drawn
 * as a connected timeline of steps.
 *
 * ## The shape, and why
 *
 * A continuous 1px rail runs down the left gutter and every step's node sits
 * on it, so a run reads as one thread of work rather than a list of unrelated
 * lines. A filled node is a top-level step; nested sub-steps get a hollow one
 * and sit further in. Colour carries state and nothing else: muted for
 * finished, the accent blue for the step in flight, red for one that failed.
 *
 * A tool call carries the tool's icon, its name, and a right-aligned count of
 * what came back. While the call is running that count is replaced by a
 * pulsing dot, because a number that is not yet true should not be on screen.
 *
 * ## Two deliberate departures from the reference design
 *
 * The reference hard-codes `rgba(255,255,255,x)` throughout. Those are written
 * here as the theme's ink tokens instead, so the tree stays legible in the
 * light theme rather than going white-on-white; in dark, which is the theme it
 * was drawn for, the values are the same.
 *
 * Expansion animates through a `grid-template-rows` 0fr→1fr transition rather
 * than an animated `max-height`. A max-height animation has to guess a
 * ceiling, and a step whose detail runs past the guess either clips or snaps
 * at the end; this measures nothing and so cannot be wrong.
 */
export function ThinkingTree({ nodes, isLive, summary }: ThinkingTreeProps) {
  const [open, setOpen] = useState(isLive);
  const wasLive = useRef(isLive);

  useEffect(() => {
    if (wasLive.current !== isLive) setOpen(isLive);
    wasLive.current = isLive;
  }, [isLive]);

  if (nodes.length === 0) return null;

  const cards = groupIntoCards(nodes);
  const failed = nodes.filter(n => n.status === 'failed').length;
  const label =
    `${nodes.length} step${nodes.length === 1 ? '' : 's'}` +
    (failed > 0 ? `, ${failed} failed` : '');

  return (
    <div className="my-2 flex flex-col gap-2 text-[13px]">
      {/* Live, the tree is open — watching the steps arrive is the point.
        * Finished, it folds to one line: the answer is what the reader wants
        * then, with the work available but out of the way. */}
      {!isLive && (
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
          className="flex w-full items-center gap-2 rounded-node px-1 py-1 text-left
                     text-ink-faint transition-colors hover:text-ink-muted
                     focus-visible:outline focus-visible:outline-1
                     focus-visible:outline-line"
        >
          <Check size={13} className="shrink-0 text-thinking" />
          <span className="flex-1 truncate">Worked through {label}</span>
          {summary && (
            <span className="shrink-0 tabular-nums text-ink-faint">{summary}</span>
          )}
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </button>
      )}

      {open &&
        cards.map((card, cardIdx) => (
          <div
            key={`${card.key}-${cardIdx}`}
            className="rounded-node border border-line/70 bg-surface-raised/40 py-1"
          >
            {card.nodes.map((node, idx) => (
              <ActivityRow
                key={node.id}
                node={node}
                isFirst={idx === 0}
                isLast={idx === card.nodes.length - 1}
              />
            ))}
          </div>
        ))}
    </div>
  );
}

/** Consecutive nodes sharing a `group` key become one card. */
function groupIntoCards(nodes: ThinkingNode[]): { key: string; nodes: ThinkingNode[] }[] {
  const cards: { key: string; nodes: ThinkingNode[] }[] = [];
  for (const node of nodes) {
    const key = node.group ?? 'default';
    const last = cards[cards.length - 1];
    if (last && last.key === key) last.nodes.push(node);
    else cards.push({ key, nodes: [node] });
  }
  return cards;
}

/** A step the model spent on thinking rather than on a tool. */
function isThought(node: ThinkingNode): boolean {
  const l = node.label.trim().toLowerCase();
  return l === 'think' || l === 'thinking' || l.startsWith('thinking ');
}

function RowIcon({ node }: { node: ThinkingNode }) {
  if (node.status === 'failed') {
    return <AlertTriangle size={13} />;
  }
  if (isThought(node)) return <Brain size={13} />;
  switch (node.icon) {
    case 'search':
      return <FileSearch size={13} />;
    case 'link':
      return <Globe size={13} />;
    case 'tool':
      return <Wrench size={13} />;
    default:
      // The node on the rail. Filled for a top-level step, and the accent
      // blue while that step is the one in flight.
      return (
        <span
          className={
            'block h-[7px] w-[7px] rounded-full ' +
            (node.status === 'running' ? 'animate-pulse bg-thinking' : 'bg-ink-faint')
          }
        />
      );
  }
}

function ActivityRow({
  node,
  isFirst,
  isLast,
}: {
  node: ThinkingNode;
  isFirst: boolean;
  isLast: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const children = node.children ?? [];
  const expandable = children.length > 0 || Boolean(node.detail);
  const running = node.status === 'running';
  const failed = node.status === 'failed';

  const body = (
    <>
      {/* Gutter: the rail, and the node sitting on it. The rail stops at the
        * first node's centre and at the last one's, so an unterminated line
        * never suggests a step that is not coming. */}
      <span className="relative flex w-5 shrink-0 justify-center self-stretch">
        <span
          aria-hidden="true"
          className="absolute left-1/2 w-px -translate-x-1/2 bg-line"
          style={{ top: isFirst ? '50%' : 0, bottom: isLast && !expanded ? '50%' : 0 }}
        />
        <span
          className={
            'relative z-10 flex h-4 w-4 items-center justify-center rounded-full bg-surface ' +
            (failed ? 'text-red-500' : running ? 'text-thinking' : 'text-ink-faint')
          }
        >
          <RowIcon node={node} />
        </span>
      </span>

      <span
        className={
          'flex-1 truncate text-left ' +
          (failed ? 'text-red-400' : running ? 'text-ink' : 'text-ink-muted')
        }
      >
        {node.label}
      </span>

      {node.badge && (
        <span
          title={node.badge}
          className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full
                     border border-line text-[9px] uppercase text-ink-faint"
        >
          {node.badge.slice(0, 1)}
        </span>
      )}

      {/* The count of what a tool returned — "3 pages". While the call is in
        * flight a pulsing dot stands in its place: the number is not known
        * yet, and a guessed or stale one would be worse than none. */}
      {running ? (
        <span
          aria-label="running"
          className="h-[6px] w-[6px] shrink-0 animate-pulse rounded-full bg-thinking"
        />
      ) : (
        node.meta && (
          <span className="shrink-0 text-[12px] tabular-nums text-ink-faint">
            {node.meta}
          </span>
        )
      )}

      <span className="flex w-4 shrink-0 justify-end text-ink-faint">
        {expandable && (expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />)}
      </span>
    </>
  );

  const rowClass =
    'flex w-full items-center gap-2 px-2 py-[5px] text-[13px] ' +
    'transition-[filter,background-color] hover:brightness-110 ' +
    'motion-reduce:transition-none';

  return (
    <div>
      {expandable ? (
        <button
          type="button"
          aria-expanded={expanded}
          onClick={() => setExpanded(e => !e)}
          className={
            rowClass +
            ' cursor-pointer text-left hover:bg-surface-hover/40' +
            ' focus-visible:outline focus-visible:outline-1 focus-visible:outline-line'
          }
        >
          {body}
        </button>
      ) : (
        <div className={rowClass}>{body}</div>
      )}

      {/* 0fr → 1fr: a real height transition with nothing measured. */}
      <div
        className="grid transition-[grid-template-rows] duration-200 ease-out
                   motion-reduce:transition-none"
        style={{ gridTemplateRows: expanded && expandable ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
          <div className="ml-5 border-l border-line pl-3">
            {node.detail && (
              <p className="my-1 whitespace-pre-wrap text-[12.5px] leading-relaxed text-ink-faint">
                {node.detail}
              </p>
            )}
            {children.map(child => (
              <div
                key={child.id}
                className="flex items-center gap-2 py-[3px] text-[12.5px] text-ink-faint"
              >
                {/* Hollow, against the parent's filled node. */}
                {child.status === 'running' ? (
                  <span className="h-[6px] w-[6px] shrink-0 animate-pulse rounded-full bg-thinking" />
                ) : child.icon === 'link' ? (
                  <Globe size={11} className="shrink-0" />
                ) : (
                  <span
                    aria-hidden="true"
                    className="h-[6px] w-[6px] shrink-0 rounded-full border border-ink-faint"
                  />
                )}
                <span className="flex-1 truncate">{child.label}</span>
                {child.meta && <span className="shrink-0 tabular-nums">{child.meta}</span>}
                {child.href && (
                  <a
                    href={child.href}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`Open ${child.label}`}
                    className="shrink-0 text-ink-faint hover:text-ink"
                  >
                    <ExternalLink size={11} />
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
