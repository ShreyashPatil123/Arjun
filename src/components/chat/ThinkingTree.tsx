import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, CheckCircle2, Globe, ExternalLink } from 'lucide-react';
import styles from './ChatSurface.module.css';

export interface ThinkingNode {
  id: string;
  label: string;
  icon?: 'search' | 'link' | 'tool' | 'none';
  meta?: string;
  status?: 'running' | 'done' | 'failed';
  children?: ThinkingNode[];
  href?: string;
}

interface ThinkingTreeProps {
  nodes: ThinkingNode[];
  isLive: boolean;
}

export function ThinkingTree({ nodes, isLive }: ThinkingTreeProps) {
  const [open, setOpen] = useState(true);
  const allDone = !isLive && nodes.every(
    n => n.status !== 'running' && !(n.children?.some(c => c.status === 'running'))
  );

  return (
    <div className={styles.thinkingTree}>
      <button
        type="button"
        className={styles.thinkingTreeHeader}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        {isLive ? (
          <Loader2 size={13} className={styles.spin} />
        ) : allDone ? (
          <CheckCircle2 size={13} className={styles.thinkingTreeDoneIcon} />
        ) : (
          <Loader2 size={13} className={styles.spin} />
        )}
        <span className={styles.thinkingTreeTitle}>Thinking</span>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>

      {open && (
        <div className={styles.thinkingTreeBody}>
          {nodes.map((node, idx) => (
            <ThinkingNodeRow key={node.id} node={node} isLast={idx === nodes.length - 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function ThinkingNodeRow({ node, isLast }: { node: ThinkingNode; isLast: boolean }) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = Boolean(node.children && node.children.length > 0);

  return (
    <div className={styles.thinkingNode}>
      {/* Vertical connector line */}
      <div className={styles.thinkingNodeLineWrap}>
        <div className={`${styles.thinkingNodeLine} ${isLast ? styles.thinkingNodeLineLast : ''}`} />
        <div className={styles.thinkingNodeDot} />
      </div>

      <div className={styles.thinkingNodeContent}>
        <button
          type="button"
          className={styles.thinkingNodeHeader}
          onClick={() => hasChildren && setExpanded(e => !e)}
          aria-expanded={expanded}
        >
          {node.status === 'running' && <Loader2 size={11} className={styles.spin} />}
          {node.status === 'done' && <CheckCircle2 size={11} className={styles.thinkingTreeDoneIcon} />}
          {node.status === 'failed' && <span className={styles.thinkingNodeFail}>✕</span>}
          {node.icon === 'search' && <Globe size={11} />}
          {node.icon === 'link' && <ExternalLink size={11} />}
          {node.icon === 'none' && <span className={styles.thinkingNodeBullet}>•</span>}

          <span className={styles.thinkingNodeLabel}>{node.label}</span>

          {node.meta && <span className={styles.thinkingNodeMeta}>{node.meta}</span>}

          {hasChildren && (expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />)}
        </button>

        {hasChildren && expanded && (
          <div className={styles.thinkingNodeChildren}>
            {node.children!.map((child, cIdx) => (
              <div key={child.id} className={styles.thinkingChildRow}>
                <div className={styles.thinkingChildLineWrap}>
                  {!isLast && <div className={styles.thinkingChildLine} />}
                  <div className={styles.thinkingChildDot} />
                </div>
                <div className={styles.thinkingChildContent}>
                  {child.status === 'running' && <Loader2 size={10} className={styles.spin} />}
                  {child.status === 'done' && <CheckCircle2 size={10} className={styles.thinkingTreeDoneIcon} />}
                  {child.icon === 'link' && <ExternalLink size={10} />}
                  {child.icon === 'search' && <Globe size={10} />}
                  <span className={styles.thinkingChildLabel}>{child.label}</span>
                  {child.meta && <span className={styles.thinkingChildMeta}>{child.meta}</span>}
                  {child.href && (
                    <a href={child.href} target="_blank" rel="noreferrer" className={styles.thinkingChildLink}>
                      ↗
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
