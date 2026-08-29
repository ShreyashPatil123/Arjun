import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ChevronDown, Clock, BookOpen, Boxes, ShieldCheck, Activity, HeartPulse, Cpu, Settings, Plus, UserRound, LogOut, MessageSquare,
} from 'lucide-react';
import {
  governanceService,
  ROLE_LABELS,
  type Session,
} from '../../services/governance.service';
import {
  agentService,
  type Conversation,
} from '../../services/agent.service';
import { relativeTime } from '../../utils/relativeTime';
import styles from './AppMenu.module.css';

/** A menu entry. `shortcut` is the modifier-less key; the modifier is rendered
 *  per-platform so Windows shows Ctrl rather than the mac command glyph. */
interface MenuItem {
  label: string;
  icon: React.ReactNode;
  path: string;
  shortcut: string;
  /** Drawn above this item, to separate creating work from reviewing it. */
  dividerBefore?: boolean;
}

/* Navigation is shaped by what PS 26117 has to make visible, not by a generic
 * assistant shell. Every judging criterion needs somewhere to live:
 * Approvals covers the human-in-the-loop gate, and Audit & Network is where
 * the zero-egress claim is actually demonstrated rather than asserted.
 *
 * History lives here, in the dropdown, rather than as a separate "Conversations"
 * route: the user already had a dedicated sidebar before, and replacing it
 * with another full-page list would just re-create the same prominence.
 * Recent chats are surfaced as a short list at the top of this menu. */
const ITEMS: MenuItem[] = [
  { label: 'New Chat',         icon: <Plus size={17} />,         path: '/',           shortcut: 'N' },
  { label: 'Tasks',            icon: <Clock size={17} />,        path: '/tasks',      shortcut: 'H' },
  { label: 'Knowledge',        icon: <BookOpen size={17} />,     path: '/knowledge',  shortcut: 'K' },
  { label: 'Models',           icon: <Boxes size={17} />,        path: '/models',     shortcut: 'M' },
  { label: 'Approvals',        icon: <ShieldCheck size={17} />,  path: '/approvals',  shortcut: 'R' },
  { label: 'Audit & Network',  icon: <Activity size={17} />,     path: '/audit',      shortcut: 'L' },
  { label: 'Health',           icon: <HeartPulse size={17} />,   path: '/health',     shortcut: 'J' },
  { label: 'Model Health',     icon: <Cpu size={17} />,          path: '/model-health', shortcut: 'O', dividerBefore: true },
  { label: 'SIH Demo',         icon: <Plus size={17} />,         path: '/demo',       shortcut: 'D' },
  { label: 'SIH Dashboard',    icon: <Activity size={17} />,     path: '/sih',        shortcut: 'P' },
  { label: 'Settings',         icon: <Settings size={17} />,     path: '/settings',   shortcut: 'A' },
];

/** Cap on the number of recent conversations listed in the dropdown. More
 *  than this and the menu starts feeling like a second sidebar. */
const RECENT_CONVERSATIONS_LIMIT = 6;

/** True on macOS, where the modifier is rendered as the command glyph. */
const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);
const MOD_LABEL = IS_MAC ? '\u2318' : 'Ctrl';

export const AppMenu = () => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [session, setSession] = useState<Session | null>(null);
  const [recent, setRecent] = useState<Conversation[]>([]);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void (async () => {
      try {
        setSession(await governanceService.currentSession());
      } catch {
        // Backend unavailable in a browser-only dev run. The menu still
        // navigates; it just cannot name who is signed in.
      }
    })();
  }, []);

  // Conversation history is fetched lazily so opening the menu does not pay
  // for the network round-trip on every page load. A failed read is silent:
  // the menu still works, it just shows no recent chats.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void (async () => {
      try {
        const list = await agentService.listConversations();
        if (cancelled) return;
        setRecent(list.slice(0, RECENT_CONVERSATIONS_LIMIT));
      } catch {
        if (cancelled) return;
        setRecent([]);
      }
    })();
    return () => { cancelled = true; };
  }, [open]);

  // Signing out reloads rather than clearing local state, so every screen is
  // rebuilt for the next person instead of inheriting the last one's data.
  const signOut = async () => {
    try {
      await governanceService.signOut();
    } finally {
      window.location.reload();
    }
  };

  const go = useCallback((path: string) => {
    setOpen(false);
    navigate(path);
  }, [navigate]);

  // Opening a chat from history: pin the conversation id so Workbench
  // reattaches to it on mount. Removing the key (a no-op here, since the
  // key is set by useConversation) would only matter if the user wanted a
  // brand-new chat, which is "New Chat" — useConversation already handles
  // that on first mount when no key is present.
  const openConversation = useCallback((id: string) => {
    setOpen(false);
    try { sessionStorage.setItem('arjun.conversation.last', id); } catch { /* noop */ }
    navigate('/');
  }, [navigate]);

  // Close on Escape, and on a click landing outside the menu. Both listeners
  // are only attached while open so the app is not paying for them at rest.
  useEffect(() => {
    if (!open) return;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    const onPointer = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };

    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onPointer);
    };
  }, [open]);

  // Global shortcuts. These fire whether or not the menu is open, so the menu
  // is a discoverability aid rather than the only way in.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
      const hit = ITEMS.find(i => i.shortcut.toLowerCase() === e.key.toLowerCase());
      if (!hit) return;
      e.preventDefault();
      go(hit.path);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [go]);

  return (
    <div className={styles.root} ref={rootRef}>
      <button
        className={styles.trigger}
        onClick={() => setOpen(o => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className={styles.wordmark}>ARJUN</span>
        <ChevronDown size={15} className={open ? styles.chevronOpen : styles.chevron} />
      </button>

      {open && (
        <div className={styles.panel} role="menu">
          {/* Who is acting. Every permission check downstream is against this
            * account, so it belongs at the top of the menu rather than buried
            * in settings. */}
          <div className={styles.account}>
            <span className={styles.accountIcon}><UserRound size={16} /></span>
            <span className={styles.accountText}>
              <span className={styles.accountName}>
                {session ? session.user.displayName : 'Not signed in'}
              </span>
              <span className={styles.accountRoles}>
                {session
                  ? session.user.roles.map(r => ROLE_LABELS[r] ?? r).join(' · ')
                  : 'No permissions are held'}
              </span>
            </span>
            {session && (
              <button
                className={styles.accountSwitch}
                onClick={() => void signOut()}
                aria-label="Sign out"
              >
                <LogOut size={13} />
              </button>
            )}
          </div>

          {/* Recent chats live here, not in a side rail. Cap is small
            * (RECENT_CONVERSATIONS_LIMIT) so the dropdown stays a dropdown. */}
          {recent.length > 0 && (
            <>
              <div className={styles.sectionHeader}>
                <MessageSquare size={12} />
                <span>Recent chats</span>
              </div>
              {recent.map(c => (
                <button
                  key={c.id}
                  className={styles.recentItem}
                  role="menuitem"
                  onClick={() => openConversation(c.id)}
                  title={c.title}
                >
                  <span className={styles.recentTitle}>{c.title || 'Untitled chat'}</span>
                  <span className={styles.recentTime}>{relativeTime(c.lastActivityAt)}</span>
                </button>
              ))}
              <div className={styles.divider} />
            </>
          )}

          {ITEMS.map(item => (
            <React.Fragment key={item.path + item.label}>
              {item.dividerBefore && <div className={styles.divider} />}
              <button className={styles.item} role="menuitem" onClick={() => go(item.path)}>
                <span className={styles.itemIcon}>{item.icon}</span>
                <span className={styles.itemLabel}>{item.label}</span>
                <span className={styles.itemShortcut}>{MOD_LABEL} {item.shortcut}</span>
              </button>
            </React.Fragment>
          ))}
        </div>
      )}
    </div>
  );
};
