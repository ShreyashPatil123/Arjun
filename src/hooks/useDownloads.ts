import { useCallback, useEffect, useRef, useState } from 'react';
import {
  getActiveDownloads,
  listenDownloadProgress,
  pauseModelDownload,
  cancelModelDownload,
} from '../services/download.service';
import type { DownloadStatus, DownloadTask, DownloadProgressPayload } from '../types/download';

/**
 * One download, in the shape the UI needs.
 *
 * The backend reports two different shapes — a `DownloadTask` when listing what
 * is already running, and a `DownloadProgressPayload` on every progress tick.
 * Normalising both here means the progress bar has one thing to render, and the
 * percentage is computed in one place rather than in each component.
 */
export interface DownloadView {
  taskId: string;
  modelId: string;
  modelName: string;
  quantization: string;
  downloadedBytes: number;
  totalBytes: number;
  /** 0–100. */
  percent: number;
  speedFormatted: string;
  etaSeconds: number | null;
  status: DownloadStatus;
  error: string | null;
}

const ACTIVE: DownloadStatus[] = ['Resolving', 'Queued', 'Downloading', 'Paused', 'Verifying'];

export function isActive(status: DownloadStatus): boolean {
  return ACTIVE.includes(status);
}

/** Bytes per second as something readable, e.g. `12.4 MB/s`. */
export function formatSpeed(bytesPerSecond: number): string {
  if (!bytesPerSecond || bytesPerSecond <= 0) return '';
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
  let value = bytesPerSecond;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** Seconds as a short human duration, e.g. `2m 5s`. */
export function formatEta(seconds: number | null): string {
  if (seconds === null || seconds === undefined || seconds < 0) return '';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function fromTask(task: DownloadTask): DownloadView {
  return {
    taskId: task.id,
    modelId: task.modelId,
    modelName: task.modelName,
    quantization: task.quantization,
    downloadedBytes: task.downloadedBytes,
    totalBytes: task.totalBytes,
    // Guarded: total is 0 while the size is still being resolved, and dividing
    // by it would put NaN into the bar's width.
    percent: task.totalBytes > 0 ? (task.downloadedBytes / task.totalBytes) * 100 : 0,
    speedFormatted: formatSpeed(task.speedBps),
    etaSeconds: task.etaSeconds,
    status: task.status,
    error: task.error,
  };
}

function fromProgress(p: DownloadProgressPayload, previous?: DownloadView): DownloadView {
  return {
    taskId: p.taskId,
    modelId: p.modelId,
    // Progress events carry no display name, so the one from the initial
    // listing is kept rather than falling back to a raw repository id.
    modelName: previous?.modelName ?? p.modelId,
    quantization: p.quantization,
    downloadedBytes: p.downloadedBytes,
    totalBytes: p.totalBytes,
    percent: p.progressPercent,
    speedFormatted: p.speedFormatted || formatSpeed(p.speedBps),
    etaSeconds: p.etaSeconds,
    status: p.status,
    error: p.error,
  };
}

/**
 * Live download state, shared by Discover and Storage.
 *
 * `onCompleted` fires when a download finishes, so a page can refresh what it
 * shows as installed without polling for it.
 */
export function useDownloads(onCompleted?: (view: DownloadView) => void) {
  const [downloads, setDownloads] = useState<Record<string, DownloadView>>({});

  // Held in a ref so a caller passing an inline arrow function does not tear
  // down and re-create the event subscription on every render.
  const completedRef = useRef(onCompleted);
  completedRef.current = onCompleted;

  const refresh = useCallback(async () => {
    try {
      const tasks = await getActiveDownloads();
      setDownloads((prev) => {
        const next = { ...prev };
        for (const task of tasks ?? []) next[task.id] = fromTask(task);
        return next;
      });
    } catch {
      // A failed listing should not blank the bars that are already on screen;
      // the event stream keeps them accurate regardless.
    }
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    void refresh();

    void listenDownloadProgress((payload) => {
      setDownloads((prev) => {
        const view = fromProgress(payload, prev[payload.taskId]);

        if (view.status === 'Completed' && prev[payload.taskId]?.status !== 'Completed') {
          completedRef.current?.(view);
        }

        // Cancelled downloads leave nothing behind, so they leave the list too.
        // Failures stay visible — a download that stopped needs to say why.
        if (view.status === 'Cancelled') {
          const { [payload.taskId]: _removed, ...rest } = prev;
          return rest;
        }

        return { ...prev, [payload.taskId]: view };
      });
    }).then((fn) => {
      if (cancelled) {
        fn();
        return;
      }
      unlisten = fn;
    });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [refresh]);

  const list = Object.values(downloads).sort((a, b) => a.modelName.localeCompare(b.modelName));

  const pause = useCallback(async (taskId: string) => {
    await pauseModelDownload(taskId);
  }, []);

  // The bar is removed by the `Cancelled` event the backend sends, not here.
  // Removing it optimistically meant a cancel that failed still cleared the
  // bar, so a download could carry on with nothing on screen saying so.
  const cancel = useCallback(async (taskId: string) => {
    await cancelModelDownload(taskId);
  }, []);

  /** Whether a given model+quantization is already downloading. */
  const isDownloading = useCallback(
    (modelId: string, quantization?: string) =>
      list.some(
        (d) =>
          d.modelId === modelId &&
          isActive(d.status) &&
          (!quantization || d.quantization === quantization)
      ),
    [list]
  );

  const dismiss = useCallback((taskId: string) => {
    setDownloads((prev) => {
      const { [taskId]: _removed, ...rest } = prev;
      return rest;
    });
  }, []);

  return { downloads: list, refresh, pause, cancel, dismiss, isDownloading };
}
