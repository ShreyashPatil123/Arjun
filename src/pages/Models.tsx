import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  ArrowLeft,
  RefreshCw,
  Layers,
  Zap,
  ShieldAlert,
  Download,
  Pause,
  Play,
  XCircle,
  HardDrive,
  Trash2,
  FolderDown,
} from 'lucide-react';
import { Card, Button, Badge, Spinner } from '../components/ui';
import { useToast } from '../hooks/useToast';
import { getModelRecommendations } from '../services/recommendation.service';
import {
  startModelDownload,
  pauseModelDownload,
  cancelModelDownload,
  getActiveDownloads,
  getInstalledModels,
  deleteInstalledModel,
  getStorageSummary,
  listenDownloadProgress,
} from '../services/download.service';
import type { ModelRecommendation, FitCategory } from '../types/recommendation';
import type {
  DownloadProgressPayload,
  DownloadTask,
  InstalledModel,
  StorageSummary,
} from '../types/download';
import styles from './Models.module.css';

type ActiveTab = FitCategory | 'Storage';

export const Models: React.FC = () => {
  const navigate = useNavigate();
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
  const [activeTasks, setActiveTasks] = useState<Record<string, DownloadProgressPayload>>({});
  const [installedModels, setInstalledModels] = useState<InstalledModel[]>([]);
  const [storageSummary, setStorageSummary] = useState<StorageSummary | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('Recommended');
  const { addToast } = useToast();

  const loadRecommendations = useCallback(async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getModelRecommendations(forceRefresh);
      setRecommendations(data);
    } catch (err) {
      console.error('Failed to load model recommendations:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      addToast('error', `Failed to generate model recommendations: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  const refreshStorageAndDownloads = useCallback(async () => {
    try {
      const [installed, storage, downloads] = await Promise.all([
        getInstalledModels(),
        getStorageSummary(),
        getActiveDownloads(),
      ]);
      setInstalledModels(installed);
      setStorageSummary(storage);

      const tasksMap: Record<string, DownloadProgressPayload> = {};
      for (const t of downloads) {
        tasksMap[t.id] = {
          taskId: t.id,
          modelId: t.modelId,
          quantization: t.quantization,
          downloadedBytes: t.downloadedBytes,
          totalBytes: t.totalBytes,
          progressPercent: t.totalBytes > 0 ? (t.downloadedBytes / t.totalBytes) * 100 : 0,
          speedBps: t.speedBps,
          speedFormatted:
            t.speedBps >= 1048576
              ? `${(t.speedBps / 1048576).toFixed(1)} MB/s`
              : `${(t.speedBps / 1024).toFixed(0)} KB/s`,
          etaSeconds: t.etaSeconds,
          status: t.status,
          error: t.error,
        };
      }
      setActiveTasks(tasksMap);
    } catch (err) {
      console.error('Failed to refresh storage/downloads:', err);
    }
  }, []);

  useEffect(() => {
    loadRecommendations();
    refreshStorageAndDownloads();
  }, [loadRecommendations, refreshStorageAndDownloads]);

  // Listen to native download progress events
  useEffect(() => {
    let unlistenFn: (() => void) | null = null;
    listenDownloadProgress((payload) => {
      setActiveTasks((prev) => ({
        ...prev,
        [payload.taskId]: payload,
      }));
      if (payload.status === 'Completed') {
        addToast('success', `${payload.modelId} (${payload.quantization}) downloaded successfully!`);
        refreshStorageAndDownloads();
      } else if (payload.status === 'Failed') {
        addToast('error', `Download failed for ${payload.modelId}: ${payload.error || 'Unknown error'}`);
        refreshStorageAndDownloads();
      }
    }).then((unlisten) => {
      unlistenFn = unlisten;
    });

    return () => {
      if (unlistenFn) unlistenFn();
    };
  }, [refreshStorageAndDownloads, addToast]);

  const handleStartDownload = async (model: ModelRecommendation) => {
    try {
      addToast('info', `Resolving HuggingFace artifact for ${model.modelName} (${model.quantization})...`);
      const taskId = await startModelDownload({
        modelId: model.modelId,
        modelName: model.modelName,
        providerId: model.providerId || 'huggingface',
        quantization: model.quantization,
        format: 'GGUF',
        backend: model.backend,
      });
      addToast('success', `Download task initiated: ${model.modelName} (${model.quantization})`);
      refreshStorageAndDownloads();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast('error', `Download failed to start: ${msg}`);
    }
  };

  const handlePause = async (taskId: string) => {
    try {
      await pauseModelDownload(taskId);
      addToast('info', 'Download paused');
      refreshStorageAndDownloads();
    } catch (err) {
      addToast('error', `Failed to pause: ${String(err)}`);
    }
  };

  const handleCancel = async (taskId: string) => {
    try {
      await cancelModelDownload(taskId);
      addToast('info', 'Download cancelled & temporary files cleaned up');
      refreshStorageAndDownloads();
    } catch (err) {
      addToast('error', `Failed to cancel: ${String(err)}`);
    }
  };

  const handleDeleteModel = async (providerId: string, modelId: string, quantization: string) => {
    try {
      await deleteInstalledModel(providerId, modelId, quantization);
      addToast('success', `Deleted model ${modelId} (${quantization})`);
      refreshStorageAndDownloads();
    } catch (err) {
      addToast('error', `Failed to delete model: ${String(err)}`);
    }
  };

  const recommendedModels = recommendations.filter((r) => r.category === 'Recommended');
  const compatibleModels = recommendations.filter((r) => r.category === 'Compatible');
  const mayRunModels = recommendations.filter((r) => r.category === 'MayRun');

  const currentModels =
    activeTab === 'Recommended'
      ? recommendedModels
      : activeTab === 'Compatible'
      ? compatibleModels
      : activeTab === 'MayRun'
      ? mayRunModels
      : [];

  const formatBytes = (bytes: number) => {
    if (!bytes || bytes === 0) return '0 B';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
  };

  const formatEta = (seconds: number | null) => {
    if (seconds === null || seconds === undefined || seconds <= 0) return '';
    if (seconds < 60) return `ETA: ${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `ETA: ${mins}m ${secs}s`;
  };

  const getTaskForModel = (modelId?: string, quant?: string): DownloadProgressPayload | null => {
    if (!modelId || !quant) return null;
    const taskId = `dl_${modelId.replace(/\//g, '_')}_${quant.toLowerCase()}`;
    return activeTasks[taskId] || null;
  };

  const isModelInstalled = (modelId?: string, quant?: string): boolean => {
    if (!modelId || !quant) return false;
    return (installedModels || []).some((m) => {
      const id = m?.modelId || (m as any)?.model_id;
      const q = m?.quantization;
      return Boolean(id && q && id.toLowerCase() === modelId.toLowerCase() && q.toLowerCase() === quant.toLowerCase());
    });
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerInfo}>
          <div className={styles.titleRow}>
            <Button variant="ghost" size="sm" onClick={() => navigate('/system')} style={{ marginRight: 8 }}>
              <ArrowLeft size={16} style={{ marginRight: 4 }} />
              System Specs
            </Button>
            <Sparkles size={24} color="var(--accent)" />
            <h1 className={styles.headerTitle}>AI Model Recommendations & Downloads</h1>
          </div>
          <p className={styles.headerSubtitle}>
            Hardware-matched local LLM recommendations & native background download manager
          </p>
        </div>
        <div className={styles.headerActions}>
          <Button variant="secondary" onClick={() => { loadRecommendations(true); refreshStorageAndDownloads(); }} disabled={loading}>
            <RefreshCw size={14} className={loading ? styles.spinningIcon : ''} style={{ marginRight: 6 }} />
            {loading ? 'Analyzing...' : 'Refresh'}
          </Button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className={styles.tabsRow}>
        <button
          className={`${styles.tabBtn} ${activeTab === 'Recommended' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('Recommended')}
        >
          <CheckCircle2 size={16} color={activeTab === 'Recommended' ? 'var(--accent)' : undefined} />
          Recommended
          <span className={styles.tabBadge}>{recommendedModels.length}</span>
        </button>
        <button
          className={`${styles.tabBtn} ${activeTab === 'Compatible' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('Compatible')}
        >
          <Zap size={16} />
          Compatible
          <span className={styles.tabBadge}>{compatibleModels.length}</span>
        </button>
        <button
          className={`${styles.tabBtn} ${activeTab === 'MayRun' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('MayRun')}
        >
          <AlertTriangle size={16} />
          May Run
          <span className={styles.tabBadge}>{mayRunModels.length}</span>
        </button>
        <button
          className={`${styles.tabBtn} ${activeTab === 'Storage' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('Storage')}
        >
          <HardDrive size={16} />
          Storage & Installed
          <span className={styles.tabBadge}>{installedModels.length}</span>
        </button>
      </div>

      {/* Storage Tab View */}
      {activeTab === 'Storage' && (
        <div className={styles.storageCard}>
          <div className={styles.storageHeader}>
            <div>
              <h3 style={{ margin: 0 }}>Installed Models & Disk Storage</h3>
              <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: 'var(--text-secondary)' }}>
                Storage Location: {storageSummary?.modelsDirectory || 'Loading...'}
              </p>
            </div>
            {storageSummary && (
              <Badge variant="default">
                Free Space: {formatBytes(storageSummary.availableDiskSpaceBytes)} / {formatBytes(storageSummary.totalDiskSpaceBytes)}
              </Badge>
            )}
          </div>

          {installedModels.length === 0 ? (
            <div className={styles.emptyState}>
              <FolderDown size={36} />
              <p>No models downloaded yet. Select a recommended model above to download!</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {installedModels.map((m) => (
                <div key={m.id} className={styles.storageItem}>
                  <div>
                    <strong style={{ color: 'var(--text-primary)' }}>{m.modelName}</strong> ({m.quantization})
                    <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: 2 }}>
                      Provider: {m.providerId} · Format: {m.format} ({m.backend})
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: 2 }}>
                      Path: {m.filePath} · Size: {formatBytes(m.sizeBytes)}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <Badge variant="success">Ready</Badge>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteModel(m.providerId, m.modelId, m.quantization)}
                    >
                      <Trash2 size={14} color="var(--error)" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Recommendation Disclaimer Banner */}
      {activeTab === 'MayRun' && (
        <div className={styles.disclaimerBanner}>
          <AlertTriangle size={18} />
          <span>
            These models may run on your system, but performance and stability are not guaranteed due to tight memory headroom or heavy offloading.
          </span>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className={styles.disclaimerBanner} style={{ borderColor: 'var(--error)', background: 'rgba(239,68,68,0.1)' }}>
          <ShieldAlert size={20} color="var(--error)" />
          <div style={{ flex: 1 }}>
            <strong style={{ color: 'var(--error)' }}>Failed to calculate recommendations</strong>
            <p style={{ margin: '4px 0 0 0', fontSize: '12px' }}>{error}</p>
          </div>
          <Button variant="secondary" size="sm" onClick={() => loadRecommendations()}>
            Retry Calculation
          </Button>
        </div>
      )}

      {/* Cards Grid */}
      {activeTab !== 'Storage' && (
        loading ? (
          <div className={styles.emptyState}>
            <Spinner size="lg" />
            <p>Analyzing system hardware & evaluating catalog model configurations...</p>
          </div>
        ) : currentModels.length === 0 ? (
          <div className={styles.emptyState}>
            <Layers size={40} />
            <h3>No models found in this category</h3>
            <p>Try checking the other tabs to see compatible models for your hardware configuration.</p>
          </div>
        ) : (
          <div className={styles.cardsGrid}>
            {currentModels.map((model) => {
              const downloadTask = getTaskForModel(model.modelId, model.quantization);
              const installed = isModelInstalled(model.modelId, model.quantization);

              return (
                <Card key={model.modelId} padding="lg">
                  <div className={styles.cardHeaderRow}>
                    <div className={styles.modelIdentity}>
                      <span className={styles.modelName}>{model.modelName}</span>
                      <span className={styles.modelFamily}>{model.modelFamily}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      <Badge variant={model.confidence === 'High' ? 'success' : model.confidence === 'Medium' ? 'warning' : 'default'}>
                        {model.confidence} Confidence
                      </Badge>
                      <span className={styles.archBadge}>{model.architecture}</span>
                    </div>
                  </div>

                  <div className={styles.specsList}>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>Download Size</span>
                      <span className={styles.specVal} style={{ fontWeight: 600, color: 'var(--accent)' }}>
                        {formatBytes(model.downloadSizeBytes || 0)}
                      </span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>Quantization</span>
                      <span className={styles.specVal}>{model.quantization} ({model.quantizationBitsPerWeight.toFixed(2)} bpw)</span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>Recommended Context</span>
                      <span className={styles.specVal}>{model.recommendedContext.toLocaleString()} tokens</span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>Backend</span>
                      <span className={styles.specVal}>{model.backend}</span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>Run Mode</span>
                      <span className={styles.specVal}>{model.runMode}</span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>VRAM Required</span>
                      <span className={styles.specVal}>{formatBytes(model.estimatedVramBytes)}</span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>RAM Required</span>
                      <span className={styles.specVal}>{formatBytes(model.estimatedRamBytes)}</span>
                    </div>
                    <div className={styles.specItem}>
                      <span className={styles.specLabel}>Headroom</span>
                      <span className={styles.specVal}>{model.headroomPercent.toFixed(0)}% free</span>
                    </div>
                  </div>

                  <div className={styles.explanationBox}>{model.explanation}</div>

                  {model.warnings.map((warn, idx) => (
                    <div key={idx} className={styles.warningItem}>
                      <AlertTriangle size={12} />
                      <span>{warn}</span>
                    </div>
                  ))}

                  {/* Phase 4 Download Footer */}
                  <div className={styles.cardFooter}>
                    {installed ? (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                        <Badge variant="success">✓ Installed & Ready</Badge>
                        <Button variant="ghost" size="sm" onClick={() => setActiveTab('Storage')}>
                          View in Storage
                        </Button>
                      </div>
                    ) : downloadTask && downloadTask.status !== 'Cancelled' ? (
                      <div className={styles.progressContainer}>
                        <div className={styles.progressStats}>
                          <span>Status: <strong>{downloadTask.status}</strong></span>
                          <span>{downloadTask.status === 'Resolving' ? 'Starting...' : `${downloadTask.progressPercent.toFixed(1)}%`}</span>
                        </div>
                        <div className={styles.progressBarBg}>
                          <div className={styles.progressBarFill} style={{ width: `${Math.min(100, downloadTask.progressPercent)}%` }} />
                        </div>
                        <div className={styles.progressStats}>
                          <span>
                            {downloadTask.status === 'Resolving'
                              ? 'Resolving artifact details...'
                              : downloadTask.totalBytes > 0
                              ? `${formatBytes(downloadTask.downloadedBytes)} / ${formatBytes(downloadTask.totalBytes)}`
                              : 'Connecting...'}
                          </span>
                          <span>{downloadTask.speedFormatted} {formatEta(downloadTask.etaSeconds)}</span>
                        </div>
                        {downloadTask.error && (
                          <div style={{ color: 'var(--error)', fontSize: '11px', marginTop: 4 }}>
                            {downloadTask.error}
                          </div>
                        )}
                        <div style={{ display: 'flex', gap: '8px', marginTop: 6 }}>
                          {downloadTask.status === 'Downloading' ? (
                            <Button variant="secondary" size="sm" onClick={() => handlePause(downloadTask.taskId)}>
                              <Pause size={14} style={{ marginRight: 4 }} /> Pause
                            </Button>
                          ) : downloadTask.status === 'Failed' ? (
                            <Button variant="primary" size="sm" onClick={() => handleStartDownload(model)}>
                              <RefreshCw size={14} style={{ marginRight: 4 }} /> Retry
                            </Button>
                          ) : (
                            <Button variant="secondary" size="sm" onClick={() => handleStartDownload(model)}>
                              <Play size={14} style={{ marginRight: 4 }} /> Resume
                            </Button>
                          )}
                          <Button variant="ghost" size="sm" onClick={() => handleCancel(downloadTask.taskId)}>
                            <XCircle size={14} style={{ marginRight: 4 }} /> Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <Button variant="primary" onClick={() => handleStartDownload(model)} style={{ width: '100%' }}>
                        <Download size={16} style={{ marginRight: 6 }} /> Download Model ({model.quantization})
                      </Button>
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
        )
      )}
    </div>
  );
};