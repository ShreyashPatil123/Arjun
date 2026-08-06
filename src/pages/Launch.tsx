import React, { useCallback, useEffect, useState } from 'react';
import {
  Play,
  Download,
  Square,
  RefreshCw,
  AlertTriangle,
  Activity,
  Server,
  Terminal,
  FileJson,
} from 'lucide-react';
import { Button, Spinner } from '../components/ui';
import { useToast } from '../hooks/useToast';
import {
  forgetToolProcess,
  getLaunchOverview,
  installTool,
  launchTool,
  listenToolInstallProgress,
  previewToolInstall,
  userToolsFile,
  type LaunchOverview,
  type ToolStatus,
} from '../services/launcher.service';
import styles from './Launch.module.css';

/** How often the server panel refreshes while the screen is open. */
const REFRESH_MS = 2000;

/** Relative time for "last seen", kept short enough for a table cell. */
function sinceLabel(lastSeenMs: number): string {
  const seconds = Math.max(0, Math.round((Date.now() - lastSeenMs) / 1000));
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export const Launch: React.FC = () => {
  const { addToast } = useToast();
  const [overview, setOverview] = useState<LaunchOverview | null>(null);
  const [loading, setLoading] = useState(true);
  /** Tool id currently mid-action, so only that card shows a spinner. */
  const [busyTool, setBusyTool] = useState<string | null>(null);
  const [toolsPath, setToolsPath] = useState<string>('');
  /**
   * Last polling failure, shown once in the page rather than as a toast.
   *
   * This screen refreshes every couple of seconds; toasting each failure turned
   * one broken call into an endless stack of identical notifications that hid
   * the app behind them.
   */
  const [pollError, setPollError] = useState<string | null>(null);
  /** Latest line the package manager printed, per tool, while installing. */
  const [installLine, setInstallLine] = useState<Record<string, string>>({});

  const refresh = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      setOverview(await getLaunchOverview());
      setPollError(null);
    } catch (err) {
      setPollError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh(true);
    userToolsFile().then(setToolsPath).catch(() => {});
  }, [refresh]);

  useEffect(() => {
    const pending = listenToolInstallProgress(({ toolId, line }) => {
      setInstallLine((prev) => ({ ...prev, [toolId]: line }));
    });
    return () => {
      void pending.then((unlisten) => unlisten());
    };
  }, []);

  // Keep the server panel live. Detection runs external programs, so this is
  // deliberately a poll rather than something faster.
  useEffect(() => {
    const id = setInterval(() => refresh(false), REFRESH_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const handleInstall = async (tool: ToolStatus) => {
    setBusyTool(tool.id);
    try {
      // Show exactly what will run, and get consent, before anything executes.
      const command = await previewToolInstall(tool.id);
      const approved = window.confirm(
        `Sarathi will run:\n\n${command}\n\n` +
          `This installs ${tool.name} using a package manager already on your computer. Continue?`
      );
      if (!approved) return;

      addToast('info', `Installing ${tool.name}…`);
      await installTool(tool.id);
      addToast('success', `${tool.name} installed`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusyTool(null);
      setInstallLine((prev) => {
        const next = { ...prev };
        delete next[tool.id];
        return next;
      });
    }
  };

  const handleLaunch = async (tool: ToolStatus) => {
    setBusyTool(tool.id);
    try {
      await launchTool(tool.id);
      addToast('success', `${tool.name} started and connected`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusyTool(null);
    }
  };

  const handleStop = async (tool: ToolStatus) => {
    setBusyTool(tool.id);
    try {
      await forgetToolProcess(tool.id);
      await refresh();
    } finally {
      setBusyTool(null);
    }
  };

  if (loading && !overview) {
    return (
      <div className={styles.centered}>
        <Spinner />
        <p>Checking which tools are installed…</p>
      </div>
    );
  }

  // Nothing ever loaded — show why, rather than a blank screen.
  if (!overview) {
    return (
      <div className={styles.centered}>
        <AlertTriangle size={22} />
        <p>Could not read tool status.</p>
        {pollError && <p className={styles.muted}>{pollError}</p>}
        <Button variant="secondary" size="sm" onClick={() => refresh(true)}>
          <RefreshCw size={14} /> Try again
        </Button>
      </div>
    );
  }

  const { tools, gateway, warnings, canLaunch, blockedReason } = overview;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Launch</h1>
          <p className={styles.subtitle}>
            Start a coding tool already connected to your local model.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={() => refresh(true)}>
          <RefreshCw size={14} /> Refresh
        </Button>
      </header>

      {/* A refresh failed while the page stayed usable — say so once, in place,
        * keeping the last known state on screen rather than blanking it. */}
      {pollError && (
        <div className={styles.blocked} role="status">
          <AlertTriangle size={16} />
          <span>Status may be out of date: {pollError}</span>
        </div>
      )}

      {!canLaunch && blockedReason && (
        <div className={styles.blocked} role="status">
          <AlertTriangle size={16} />
          <span>{blockedReason}</span>
        </div>
      )}

      {warnings.map((w) => (
        <div key={w} className={styles.warning} role="status">
          <AlertTriangle size={16} />
          <span>{w}</span>
        </div>
      ))}

      <section className={styles.grid}>
        {tools.map((tool) => (
          <article key={tool.id} className={styles.card}>
            <div className={styles.cardHead}>
              <div>
                <h2 className={styles.cardTitle}>
                  <Terminal size={15} /> {tool.name}
                </h2>
                <p className={styles.cardDesc}>{tool.description}</p>
              </div>
              {tool.userDefined && (
                <span className={styles.tagMuted} title="Added by you — not verified by Sarathi">
                  custom
                </span>
              )}
            </div>

            <div className={styles.cardMeta}>
              <span className={styles.tag}>
                {tool.protocol === 'anthropic' ? 'Anthropic API' : 'OpenAI API'}
              </span>
              {tool.state === 'ready' && <span className={styles.version}>{tool.version}</span>}
            </div>

            <div className={styles.cardFoot}>
              {tool.state === 'notInstalled' && (
                tool.installCommand ? (
                  <div className={styles.installing}>
                    <Button
                      size="sm"
                      onClick={() => handleInstall(tool)}
                      disabled={busyTool === tool.id}
                    >
                      {busyTool === tool.id ? <Spinner /> : <Download size={14} />} Install
                    </Button>
                    {busyTool === tool.id && installLine[tool.id] && (
                      <span className={styles.installLine} title={installLine[tool.id]}>
                        {installLine[tool.id]}
                      </span>
                    )}
                  </div>
                ) : (
                  // No usable install route: say so rather than offering a
                  // button that cannot work.
                  <span className={styles.muted}>Not installed — install it yourself first</span>
                )
              )}

              {tool.state === 'ready' && (
                <Button
                  size="sm"
                  onClick={() => handleLaunch(tool)}
                  disabled={!canLaunch || busyTool === tool.id}
                  title={canLaunch ? undefined : blockedReason ?? undefined}
                >
                  {busyTool === tool.id ? <Spinner /> : <Play size={14} />} Launch
                </Button>
              )}

              {tool.state === 'running' && (
                <>
                  <span className={styles.running}>● Running · connected</span>
                  <Button variant="ghost" size="sm" onClick={() => handleStop(tool)}>
                    <Square size={13} /> Stop tracking
                  </Button>
                </>
              )}

              {tool.state === 'problem' && (
                <span className={styles.problem}>
                  <AlertTriangle size={14} /> {tool.reason}
                </span>
              )}
            </div>
          </article>
        ))}
      </section>

      <section className={styles.server}>
        <h2 className={styles.sectionTitle}>
          <Server size={16} /> Server
        </h2>

        <div className={styles.statRow}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Status</span>
            <span className={styles.statValue}>
              {gateway.running ? '● Running' : '○ Stopped'}
            </span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Address</span>
            <span className={styles.statValue}>127.0.0.1:{gateway.port}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Requests</span>
            <span className={styles.statValue}>{gateway.totalRequests}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>In progress</span>
            <span className={styles.statValue}>{gateway.activeRequests}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Waiting</span>
            <span className={styles.statValue}>{gateway.queueDepth}</span>
          </div>
        </div>

        <h3 className={styles.subheading}>
          <Activity size={14} /> Connected tools
        </h3>
        {gateway.clients.length === 0 ? (
          <p className={styles.muted}>
            Nothing has connected yet. Launch a tool above, then ask it something.
          </p>
        ) : (
          <table className={styles.clients}>
            <thead>
              <tr>
                <th>Tool</th>
                <th>Using</th>
                <th>Requests</th>
                <th>Last seen</th>
              </tr>
            </thead>
            <tbody>
              {gateway.clients.map((c) => (
                <tr key={`${c.label}-${c.protocol}`}>
                  <td>{c.label}</td>
                  <td>{c.protocol}</td>
                  <td>{c.requests}</td>
                  <td>{sinceLabel(c.lastSeenMs)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {toolsPath && (
        <p className={styles.footnote}>
          <FileJson size={13} /> Add your own tools by editing <code>{toolsPath}</code>
        </p>
      )}
    </div>
  );
};
