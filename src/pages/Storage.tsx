import React, { useCallback, useEffect, useState } from 'react';
import {
  HardDrive,
  Trash2,
  RefreshCw,
  Power,
  Play,
  AlertTriangle,
  Download,
  Star,
} from 'lucide-react';
import { Button, Spinner, DownloadBar } from '../components/ui';
import { Can } from '../components/auth/Can';
import { useToast } from '../hooks/useToast';
import { useDownloads } from '../hooks/useDownloads';
import {
  getInstalledModels,
  deleteInstalledModel,
  getStorageSummary,
} from '../services/download.service';
import { getInferenceStatus, loadInstalledModel, unloadActiveModel } from '../services/ai.service';
import { formatSize } from '../services/catalog.service';
import {
  registryService,
  type OrchestratorModelSelection,
} from '../services/registry.service';
import styles from './Storage.module.css';

/**
 * Storage — manage what is on disk.
 *
 * Deliberately does *not* recommend or download models; that belongs in
 * Discover. Having both meant two screens that each did half the job and
 * shared a confusing name.
 */
export const Storage: React.FC = () => {
  const { addToast } = useToast();
  const [models, setModels] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [orchestrator, setOrchestrator] = useState<OrchestratorModelSelection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [installed, sum, status, configuredOrchestrator] = await Promise.all([
        getInstalledModels(),
        getStorageSummary().catch(() => null),
        getInferenceStatus().catch(() => null),
        registryService.getOrchestratorModel().catch(() => null),
      ]);
      setModels(installed ?? []);
      setSummary(sum);
      setLoadedModelId(status?.model?.modelId ?? null);
      setOrchestrator(configuredOrchestrator);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  // A finished download becomes an installed model, so the list below it has to
  // catch up on its own — otherwise the bar says "Finished" while the model is
  // nowhere to be seen until a manual refresh.
  const onDownloadCompleted = useCallback(
    (d: { modelName: string }) => {
      addToast('success', `${d.modelName} is ready to use`);
      void refresh();
    },
    [addToast, refresh]
  );

  const { downloads, pause, resume, cancel, dismiss } = useDownloads(onDownloadCompleted);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDeleteModel = async (m: any) => {
    // Deleting frees gigabytes and cannot be undone, so it is confirmed and the
    // size is named — "are you sure?" alone does not convey what is at stake.
    const ok = window.confirm(
      `Delete ${m.modelName}?\n\nThis frees ${formatSize(m.sizeBytes)}. ` +
        `You would need to download it again to use it.`
    );
    if (!ok) return;

    setBusy(m.modelId);
    try {
      await deleteInstalledModel(m.providerId, m.modelId, m.quantization);
      addToast('success', `${m.modelName} deleted`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  // Loading reads gigabytes into VRAM and takes a while, so the button reports
  // progress and every other model's button locks — two concurrent loads would
  // fight over the same VRAM budget.
  const handleLoad = async (m: any) => {
    setBusy(m.modelId);
    try {
      await loadInstalledModel(m.providerId, m.modelId, m.quantization);
      addToast('success', `${m.modelName} loaded — the gateway can serve requests`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  const handleUnload = async () => {
    try {
      await unloadActiveModel();
      addToast('info', 'Model unloaded — the gateway has nothing to serve until you load one');
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    }
  };

  const handleSetOrchestrator = async (m: any) => {
    const busyKey = `orchestrator:${m.id}`;
    setBusy(busyKey);
    try {
      const selected = await registryService.setOrchestratorModel({
        providerId: m.providerId,
        modelId: m.modelId,
        quantization: m.quantization,
      });
      setOrchestrator(selected);
      addToast(
        'success',
        `${m.modelName} is the orchestrator and will auto-load on the GPU at startup`
      );
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  if (loading) {
    return (
      <div className={styles.centered}>
        <Spinner />
        <p>Reading what is on disk…</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Storage</h1>
          <p className={styles.subtitle}>
            Models on this computer. To find new ones, use Discover.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={refresh}>
          <RefreshCw size={14} /> Refresh
        </Button>
      </header>

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {summary && (
        <div className={styles.summary}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Models</span>
            <span className={styles.statValue}>{models.length}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Used by models</span>
            <span className={styles.statValue}>{formatSize(summary.totalModelsBytes ?? 0)}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Free on disk</span>
            <span className={styles.statValue}>{formatSize(summary.availableDiskSpaceBytes ?? 0)}</span>
          </div>
        </div>
      )}

      {downloads.length > 0 && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>
            <Download size={15} /> Downloading
            <span className={styles.count}>{downloads.length}</span>
          </h2>
          <div className={styles.downloads}>
            {downloads.map((d) => (
              <DownloadBar
                key={d.taskId}
                download={d}
                onPause={(id) => void pause(id)}
                onResume={(id) => void resume(id)}
                onCancel={(id) => void cancel(id)}
                onDismiss={dismiss}
              />
            ))}
          </div>
        </section>
      )}

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <HardDrive size={15} /> Installed
        </h2>

        {models.length === 0 && (
          <p className={styles.empty}>
            Nothing installed yet. Open <strong>Discover</strong> to find a model that fits your
            hardware.
          </p>
        )}

        {models.map((m: any) => {
          const isLoaded = loadedModelId === m.modelId;
          const isOrchestrator =
            orchestrator?.providerId === m.providerId &&
            orchestrator?.modelId === m.modelId &&
            orchestrator?.quantization === m.quantization;
          return (
            <article key={`${m.modelId}-${m.quantization}`} className={styles.model}>
              <div className={styles.modelHead}>
                <div className={styles.modelInfo}>
                  <h3 className={styles.modelName}>
                    {m.modelName} <span className={styles.quant}>{m.quantization}</span>
                    {isOrchestrator && (
                      <span className={styles.orchestrator}>★ Orchestrator</span>
                    )}
                  </h3>
                  <p className={styles.modelMeta}>
                    {formatSize(m.sizeBytes)} · {m.providerId}
                  </p>
                </div>

                <div className={styles.modelActions}>
                  {!isOrchestrator && (
                    <Can permission="modifyPolicy">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleSetOrchestrator(m)}
                        disabled={busy !== null || !m.isReady}
                        title="Use this exact model variant as the startup orchestrator"
                      >
                        <Star size={13} />
                        {busy === `orchestrator:${m.id}` ? 'Saving…' : 'Set as orchestrator'}
                      </Button>
                    </Can>
                  )}
                  {isLoaded ? (
                    <>
                      <span className={styles.serving}>● Serving the gateway</span>
                      <Can permission="importModel">
                        <Button variant="ghost" size="sm" onClick={handleUnload}>
                          <Power size={13} /> Unload
                        </Button>
                      </Can>
                    </>
                  ) : (
                    <Can permission="importModel">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleLoad(m)}
                        disabled={busy !== null}
                        title="Load this model so the gateway can serve it"
                      >
                        <Play size={13} /> {busy === m.modelId ? 'Loading…' : 'Load'}
                      </Button>
                    </Can>
                  )}
                  <Can permission="importModel">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteModel(m)}
                      disabled={busy === m.modelId}
                      title="Delete this model"
                    >
                      <Trash2 size={13} />
                    </Button>
                  </Can>
                </div>
              </div>

            </article>
          );
        })}
      </section>
    </div>
  );
};
