import React from 'react';
import { MessageSquare } from 'lucide-react';
import type { Conversation } from '../../services/agent.service';
import { ContextPanel } from './ContextPanel';
import type { CompactionRecord, ContextLedgerRecord } from '../../services/agent.service';
import styles from './ChatSurface.module.css';

/**
 * The chat header — title, context chip, and (later) any orchestrator
 * plan summary. Sits above the message log.
 */
export function ChatHeader({
  conversation,
  ledger,
  compactions,
  lastCompaction,
}: {
  conversation: Conversation | null;
  ledger?: ContextLedgerRecord | null;
  compactions?: number;
  lastCompaction?: CompactionRecord | null;
}) {
  return (
    <header className={styles.chatHeader}>
      <div className={styles.chatHeaderLeft}>
        <MessageSquare size={13} />
        <span className={styles.chatHeaderTitle}>
          {conversation?.title ?? 'New conversation'}
        </span>
      </div>
      <div className={styles.chatHeaderRight}>
        <ContextPanel
          ledger={ledger}
          compactions={compactions}
          lastCompaction={lastCompaction}
        />
      </div>
    </header>
  );
}
