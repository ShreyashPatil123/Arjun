import React from 'react';
import { Pause, Play, X, AlertTriangle, Check } from 'lucide-react';
import { formatEta, type DownloadView } from '../../hooks/useDownloads';
import { formatSize } from '../../services/catalog.service';
import styles from './DownloadBar.module.css';

interface DownloadBarProps {
  download: DownloadView;
  onPause?: (taskId: string) => void;
  onResume?: (taskId: string) => void;
  onCancel?: (taskId: string) => void;
  onDismiss?: (taskId: string) => void;
}

/**
 * One download, shown as a progress bar with what a person actually wants to
 * know: how far along, how fast, and how much longer.
 *
 * A bar alone is not enough — while a download is resolving its size the
 * percentage is meaningless, so the state is spelled out in words too.
 */
export function DownloadBar({ download: d, onPause, onResume, onCancel, onDismiss }: DownloadBarProps) {
  const failed = d.status === 'Failed';
  const done = d.status === 'Completed';
  // Both states pick up from the bytes already on disk, so the same control
  // serves each — nothing already transferred is fetched twice.
  const resumable = failed || d.status === 'Paused';
  // Size is unknown until the server answers, and a bar that sits at 0% looks
  // stuck. An indeterminate stripe says "working" honestly instead.
  const unknownSize = d.totalBytes === 0 && !done && !failed;
  const percent = Math.min(100, Math.max(0, d.percent));

  const detail = () => {
    if (failed) return d.error ?? 'Download failed';
    if (done) return `Finished · ${formatSize(d.totalBytes)}`;
    if (d.status === 'Paused') return `Paused · ${formatSize(d.downloadedBytes)} so far`;
    if (d.status === 'Verifying') return 'Checking the file is complete…';
    if (d.status === 'Queued') return 'Waiting to start…';
    if (unknownSize) return 'Working out the file size…';

    const parts = [`${formatSize(d.downloadedBytes)} of ${formatSize(d.totalBytes)}`];
    if (d.speedFormatted) parts.push(d.speedFormatted);
    const eta = formatEta(d.etaSeconds);
    if (eta) parts.push(`${eta} left`);
    return parts.join(' · ');
  };

  return (
    <div className={styles.row} data-status={d.status}>
      <div className={styles.top}>
        <span className={styles.name}>
          {d.modelName}
          {d.quantization && <span className={styles.quant}>{d.quantization}</span>}
        </span>

        <span className={styles.right}>
          {failed && <AlertTriangle size={13} className={styles.failIcon} />}
          {done && <Check size={13} className={styles.doneIcon} />}
          {!failed && !done && !unknownSize && (
            <span className={styles.percent}>{Math.round(percent)}%</span>
          )}

          {d.status === 'Downloading' && onPause && (
            <button
              className={styles.iconBtn}
              onClick={() => onPause(d.taskId)}
              aria-label={`Pause ${d.modelName}`}
              title="Pause"
            >
              <Pause size={13} />
            </button>
          )}

          {resumable && onResume && (
            <button
              className={styles.iconBtn}
              onClick={() => onResume(d.taskId)}
              aria-label={`${failed ? 'Retry' : 'Resume'} ${d.modelName}`}
              title={failed ? 'Retry from where it stopped' : 'Resume'}
            >
              <Play size={13} />
            </button>
          )}

          {(done || failed) && onDismiss ? (
            <button
              className={styles.iconBtn}
              onClick={() => onDismiss(d.taskId)}
              aria-label={`Dismiss ${d.modelName}`}
              title="Dismiss"
            >
              <X size={13} />
            </button>
          ) : (
            onCancel && (
              <button
                className={styles.iconBtn}
                onClick={() => onCancel(d.taskId)}
                aria-label={`Cancel ${d.modelName}`}
                title="Cancel and delete the partial file"
              >
                <X size={13} />
              </button>
            )
          )}
        </span>
      </div>

      <div
        className={`${styles.track} ${unknownSize ? styles.indeterminate : ''}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        // Omitted while indeterminate so assistive tech announces "busy"
        // rather than a percentage that does not mean anything yet.
        aria-valuenow={unknownSize ? undefined : Math.round(percent)}
        aria-label={`${d.modelName} download`}
      >
        <div className={styles.fill} style={{ width: unknownSize ? '100%' : `${percent}%` }} />
      </div>

      <p className={styles.detail}>{detail()}</p>
    </div>
  );
}
