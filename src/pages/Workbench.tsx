import React from 'react';
import { ChatSurface } from '../components/chat/ChatSurface';
import styles from './Workbench.module.css';

/**
 * The default Arjun workbench is a chat surface.
 *
 * The previous Workbench was a one-shot composer: a single prompt was
 * sent, the composer hid itself, and a full `RunView` took the screen
 * for the duration of the run. The chat surface replaces that with a
 * persistent conversation: the composer stays visible after every
 * answer, the user can send unlimited follow-ups in the same
 * conversation, and the per-run details are opened on demand by
 * clicking "View details" on an assistant message.
 *
 * The "local-only" indicator is no longer pinned to the corner — the
 * status pill in the top bar carries the same signal, and the audit
 * screen is one ARJUN-menu click away.
 */
export const Workbench = () => {
  return (
    <div className={styles.page}>
      <div className={styles.chatWrap}>
        <ChatSurface />
      </div>
    </div>
  );
};
