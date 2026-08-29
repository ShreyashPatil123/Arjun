import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';
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
 */
export const Workbench = () => {
  return (
    <div className={styles.page}>
      <div className={styles.chatWrap}>
        <ChatSurface />
      </div>
      {/* PS step 16: a visible local-only indicator. Deliberately does
        *  not claim the task *is* secure — it reports the mode, and
        *  links to the audit screen where the claim can actually be
        *  checked. */}
      <Link className={styles.localBadge} to="/audit">
        <ShieldCheck size={14} />
        <span>Work mode &middot; no external calls</span>
      </Link>
    </div>
  );
};
