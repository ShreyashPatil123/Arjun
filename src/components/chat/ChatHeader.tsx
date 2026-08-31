import React from 'react';
import { MessageSquare } from 'lucide-react';
import type { Conversation } from '../../services/agent.service';
import type { CompactionRecord, ContextLedgerRecord } from '../../services/agent.service';
import styles from './ChatHeader.module.css';

/**
 * The breadcrumb that sits in the centre of the top bar.
 *
 * Shows the conversation title and a one-line context summary
 * (message count, last activity). Compact, never wrapping. The
 * `ContextPanel` that used to live here has moved into the
 * composer's right region (see `ContextChip`).
 */
export function ChatHeader({
  conversation,
}: {
  conversation: Conversation | null;
  ledger?: ContextLedgerRecord | null;
  compactions?: number;
  lastCompaction?: CompactionRecord | null;
}) {
  if (!conversation) {
    return (
      <div className={styles.header}>
        <span className={styles.titleEmpty}>New conversation</span>
      </div>
    );
  }

  return (
    <div className={styles.header}>
      <MessageSquare size={14} className={styles.icon} />
      <span className={styles.title} title={conversation.title}>
        {conversation.title || 'New conversation'}
      </span>
      <span className={styles.sep}>·</span>
      <span className={styles.meta}>
        {conversation.messages.length} message{conversation.messages.length === 1 ? '' : 's'}
      </span>
    </div>
  );
}
