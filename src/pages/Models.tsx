import React, { useEffect, useState, useCallback, lazy, Suspense } from 'react';
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
  MessageSquare,
  Power,
  PlayCircle,
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
import {
  loadInstalledModel,
  unloadActiveModel,
  getInferenceStatus,
  restoreLastSession,
  listenInferenceStatus,
} from '../services/ai.service';
import type { ModelRecommendation, FitCategory } from '../types/recommendation';
import type {
  DownloadProgressPayload,
  DownloadTask,
  InstalledModel,
  StorageSummary,
} from '../types/download';
import type { LoadedModelInfo, InferenceStatusPayload } from '../types/ai';
import styles from './Models.module.css';
import { Chat } from './Chat';

type ActiveTab = FitCategory | 'Storage' | 'Chat';

const isSameModelId = (id1?: string | null, id2?: string | null): boolean => {
  if (!id1 || !id2) return false;
  const n1 = id1.toLowerCase().replace(/[\/_-]/g, '');
  const n2 = id2.toLowerCase().replace(/[\/_-]/g, '');
  return n1 === n2;
};

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

  const [selectedCert, setSelectedCert] = useState<any | null>(null);
  const [showExperimental, setShowExperimental] = useState<boolean>(false);

  const loadRecommendations = useCallback(async (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getModelRecommendations(forceRefresh);
      const enriched = await Promise.all(
        data.map(async (item) => {
          const cert = await import('../services/packManager.service').then(m => m.getPackageCertification(item.modelId));
          return { ...item, certification: cert };
        })
      );
      setRecommendations(enriched);
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
      // Merge active download tasks, preserving completed adapter tasks from previous state
      // so adapter cards don't flash back to unknown during refresh cycles
      setActiveTasks((prev) => {
        const merged: Record<string, DownloadProgressPayload> = {};
        // Preserve previously completed/failed adapter tasks
        for (const [id, task] of Object.entries(prev)) {
          if (id.includes('_adapter_') && (task.status === 'Completed' || task.status === 'Failed')) {
            merged[id] = task;
          }
        }
        // Overlay active tasks from backend (these always take priority)
        for (const [id, task] of Object.entries(tasksMap)) {
          merged[id] = task;
        }
        return merged;
      });
    } catch (err) {
      console.error('Failed to refresh storage/downloads:', err);
    }
  }, []);

  const [loadedModelInfo, setLoadedModelInfo] = useState<LoadedModelInfo | null>(null);
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null);

  const refreshInferenceStatus = useCallback(async () => {
    try {
      const statusPayload = await getInferenceStatus();
      setLoadedModelInfo(statusPayload.model || null);
    } catch (err) {
      console.error('Failed to get inference status:', err);
    }
  }, []);

  useEffect(() => {
    loadRecommendations();
    refreshStorageAndDownloads();
    refreshInferenceStatus().then(() => {
      restoreLastSession().then((restored) => {
        if (restored) {
          setLoadedModelInfo(restored);
        }
      }).catch(() => {});
    });
  }, [loadRecommendations, refreshStorageAndDownloads, refreshInferenceStatus]);

  useEffect(() => {
    let unlistenFn: (() => void) | null = null;
    listenInferenceStatus((payload) => {
      setLoadedModelInfo(payload.model || null);
      if (payload.status !== 'Loading') {
        setLoadingModelId(null);
      }
    }).then((unlisten) => { unlistenFn = unlisten; });

    return () => {
      if (unlistenFn) unlistenFn();
    };
  }, []);

  const handleLoadModel = async (providerId: string, modelId: string, quantization: string) => {
    console.log(`[STAGE 1 UI] handleLoadModel entered: providerId="${providerId}", modelId="${modelId}", quantization="${quantization}"`);
    setLoadingModelId(modelId);
    try {
      addToast('info', `Loading ${modelId} (${quantization}) into local runtime...`);
      const info = await loadInstalledModel(providerId, modelId, quantization);
      console.log(`[STAGE 1 UI] loadInstalledModel SUCCESS:`, info);
      setLoadedModelInfo(info);
      addToast('success', `Model ${info.modelName} loaded into RAM/VRAM successfully!`);
    } catch (err) {
      const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      console.error(`[STAGE 1 UI] loadInstalledModel FAILED:`, err);
      addToast('error', `Model Load Error: ${msg}`);
    } finally {
      setLoadingModelId(null);
    }
  };

  const handleUnloadActiveModel = async () => {
    try {
      await unloadActiveModel();
      setLoadedModelInfo(null);
      addToast('info', 'Model unloaded from memory');
    } catch (err) {
      addToast('error', `Failed to unload model: ${String(err)}`);
    }
  };

  // Listen to native download progress events
  useEffect(() => {
    let unlistenFn: (() => void) | null = null;
    listenDownloadProgress((payload) => {
      setActiveTasks((prev) => ({
        ...prev,
        [payload.taskId]: payload,
      }));

      // Only emit single package completion success toast
      if (payload.itemType === 'package_completed' && payload.status === 'Completed') {
        addToast('success', 'Model package installed successfully');
        refreshStorageAndDownloads();
      } else if (payload.status === 'Failed' && !payload.itemType) {
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

  const rawRecommended = recommendations.filter((r) => r.category === 'Recommended');
  const rawCompatible = recommendations.filter((r) => r.category === 'Compatible');
  const rawMayRun = recommendations.filter((r) => r.category === 'MayRun');

  const filterExperimental = (list: ModelRecommendation[]) => {
    if (showExperimental) return list;
    return list.filter((item) => item.certification?.tier !== 'Experimental');
  };

  const recommendedModels = filterExperimental(rawRecommended);
  const compatibleModels = filterExperimental(rawCompatible);
  const mayRunModels = filterExperimental(rawMayRun);

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

  const getInstalledModel = (modelId?: string, quant?: string): InstalledModel | null => {
    if (!modelId || !quant) return null;
    return (installedModels || []).find((m) => {
      const id = m?.modelId || (m as any)?.model_id;
      const q = m?.quantization;
      return Boolean(id && q && id.toLowerCase() === modelId.toLowerCase() && q.toLowerCase() === quant.toLowerCase());
    }) || null;
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
      <div className={styles.tabsRow} style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            className={`${styles.tabBtn} ${activeTab === 'Recommended' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('Recommended')}
            title="Saarthi Officially Certified & Verified Packages"
          >
            <CheckCircle2 size={16} color={activeTab === 'Recommended' ? 'var(--accent)' : undefined} />
            🪷 Saarthi Recommended
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
          <button
            className={`${styles.tabBtn} ${activeTab === 'Chat' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('Chat')}
            style={{
              color: loadedModelInfo ? 'var(--success)' : undefined,
            }}
          >
            <MessageSquare size={16} />
            Chat
            {loadedModelInfo && <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--success)', display: 'inline-block', marginLeft: 4 }} />}
          </button>
        </div>

        {/* Experimental Models Filter Toggle */}
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showExperimental}
            onChange={(e) => setShowExperimental(e.target.checked)}
            style={{ accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          Show Experimental Packages ⚠️
        </label>
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
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: 2, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span>Path: {m.filePath} · Size: {formatBytes(m.sizeBytes)}</span>
                      <button
                        style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: '12px', textDecoration: 'underline', padding: 0 }}
                        onClick={async () => {
                          const cert = await import('../services/packManager.service').then(srv => srv.getPackageCertification(m.modelId));
                          if (cert) {
                            setSelectedCert(cert);
                          } else {
                            addToast('info', `No official certification profile found for ${m.modelName}.`);
                          }
                        }}
                      >
                        [View Certification Profile]
                      </button>
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: 6, display: 'flex', gap: '4px', flexWrap: 'wrap', alignItems: 'center' }}>
                      <span style={{ fontWeight: 600, color: 'var(--accent)' }}>Capability Adapters:</span>
                      {[
                        { key: 'coding', label: 'Coding' },
                        { key: 'reasoning', label: 'Reasoning' },
                        { key: 'tool-calling', label: 'Tool Calling' },
                        { key: 'mathematics', label: 'Mathematics' },
                        { key: 'research', label: 'Research' },
                      ].map(({ key, label }) => {
                        const adapter = m.adapters?.[key];
                        const isInstalled = adapter?.status === 'Installed' && Boolean(adapter?.adapterFile);
                        const runtimeStatus = adapter?.adapterRuntimeStatus;
                        const isGguf = runtimeStatus === 'compatible';
                        const needsConversion = runtimeStatus === 'requires_conversion';

                        let badgeColor = 'var(--text-secondary)';
                        let badgeBg = 'var(--surface-hover)';
                        let badgeBorder = 'var(--border)';
                        let text = 'Base Native';

                        if (isInstalled) {
                          if (isGguf) {
                            badgeColor = 'var(--success)';
                            badgeBg = 'rgba(16,185,129,0.1)';
                            badgeBorder = 'var(--success)';
                            text = `✓ GGUF Ready (${adapter?.repoId || 'LoRA'})`;
                          } else if (needsConversion) {
                            badgeColor = 'var(--warning)';
                            badgeBg = 'rgba(245,158,11,0.1)';
                            badgeBorder = 'var(--warning)';
                            text = `⚠ Needs GGUF Conversion (${adapter?.repoId || 'PEFT SafeTensors'})`;
                          } else {
                            badgeColor = 'var(--warning)';
                            badgeBg = 'rgba(245,158,11,0.1)';
                            badgeBorder = 'var(--warning)';
                            text = `Installed (${adapter?.repoId || 'LoRA'})`;
                          }
                        }

                        return (
                          <span
                            key={key}
                            style={{
                              background: badgeBg,
                              padding: '2px 6px',
                              borderRadius: '4px',
                              border: `1px solid ${badgeBorder}`,
                              color: badgeColor,
                            }}
                          >
                            ⚡ {label}: {text}
                          </span>
                        );
                      })}
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    {loadedModelInfo && isSameModelId(loadedModelInfo.modelId, m.modelId) ? (
                      <>
                        <Badge variant="success">● Loaded & Ready</Badge>
                        <Button variant="primary" size="sm" onClick={() => navigate('/chat')}>
                          <MessageSquare size={14} style={{ marginRight: 4 }} /> Open Chat
                        </Button>
                        <Button variant="secondary" size="sm" onClick={handleUnloadActiveModel}>
                          <Power size={14} style={{ marginRight: 4 }} /> Unload
                        </Button>
                      </>
                    ) : loadingModelId === m.modelId ? (
                      <Button variant="secondary" size="sm" disabled>
                        <Spinner size="sm" /> Loading...
                      </Button>
                    ) : (
                      <>
                        <Badge variant="default">Not Loaded</Badge>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => handleLoadModel(m.providerId, m.modelId, m.quantization)}
                        >
                          <PlayCircle size={14} style={{ marginRight: 4 }} /> Load Model
                        </Button>
                      </>
                    )}

                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={Boolean(loadedModelInfo && isSameModelId(loadedModelInfo.modelId, m.modelId))}
                      title={loadedModelInfo && isSameModelId(loadedModelInfo.modelId, m.modelId) ? 'Unload model first before deleting' : 'Delete installed model package'}
                      onClick={() => handleDeleteModel(m.providerId, m.modelId, m.quantization)}
                    >
                      <Trash2 size={14} color={loadedModelInfo && isSameModelId(loadedModelInfo.modelId, m.modelId) ? 'var(--text-tertiary)' : 'var(--error)'} />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Saarthi Certification Profile Drawer / Modal */}
      {selectedCert && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.75)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 999,
          backdropFilter: 'blur(6px)'
        }}>
          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '12px',
            padding: '24px',
            maxWidth: '650px',
            width: '90%',
            maxHeight: '85vh',
            overflowY: 'auto',
            boxShadow: '0 20px 40px rgba(0,0,0,0.5)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ margin: 0, color: 'var(--accent)', fontSize: '20px' }}>🪷 Saarthi Certified Model Profile</h2>
              <Button variant="ghost" size="sm" onClick={() => setSelectedCert(null)}>✕ Close</Button>
            </div>

            <div style={{ background: 'var(--surface-hover)', padding: '12px 16px', borderRadius: '8px', marginBottom: '16px' }}>
              <div style={{ fontWeight: 'bold', fontSize: '16px' }}>{selectedCert.modelName || selectedCert.packageId}</div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Tier: <span style={{ color: selectedCert.tier === 'Certified' ? 'var(--success)' : 'var(--warning)', fontWeight: 'bold' }}>{selectedCert.tier === 'Certified' ? '⭐⭐⭐⭐⭐ Saarthi Certified' : '⭐⭐⭐⭐ Compatible'}</span>
                {' · '}Confidence Score: <strong>{selectedCert.confidenceScore}/100</strong>
              </div>
            </div>

            <h4 style={{ margin: '12px 0 8px 0', color: 'var(--text-primary)' }}>⚡ LoRA Capability Compatibility Matrix</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px', marginBottom: '16px' }}>
              {selectedCert.loraCapabilityMatrix && Object.entries(selectedCert.loraCapabilityMatrix).map(([cap, status]: [string, any]) => (
                <div key={cap} style={{ background: 'var(--background)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '13px' }}>
                  <span style={{ textTransform: 'capitalize', fontWeight: '500' }}>{cap.replace('_', ' ')}:</span>{' '}
                  <span style={{ color: status === 'Certified' ? 'var(--success)' : 'var(--warning)', fontWeight: 'bold' }}>{status}</span>
                </div>
              ))}
            </div>

            <h4 style={{ margin: '12px 0 8px 0', color: 'var(--text-primary)' }}>📊 17-Point Benchmark Scores</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px', marginBottom: '16px' }}>
              {selectedCert.numericScores && Object.entries(selectedCert.numericScores).map(([k, v]: [string, any]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', background: 'var(--background)', borderRadius: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{k.replace(/_/g, ' ')}:</span>
                  <strong style={{ color: v >= 90 ? 'var(--success)' : 'var(--warning)' }}>{v}/100</strong>
                </div>
              ))}
            </div>

            <h4 style={{ margin: '12px 0 8px 0', color: 'var(--text-primary)' }}>🛡 Cryptographic Provenance</h4>
            <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', background: 'var(--background)', padding: '10px', borderRadius: '6px', fontFamily: 'monospace', wordBreak: 'break-all' }}>
              <div>Created By: {selectedCert.provenance?.createdBy}</div>
              <div>Certified By: {selectedCert.provenance?.certifiedBy}</div>
              <div>Runner Version: {selectedCert.provenance?.runnerVersion}</div>
              <div>Profile Hash: {selectedCert.provenance?.profileHash}</div>
            </div>

            <div style={{ marginTop: '20px', textAlign: 'right' }}>
              <Button variant="primary" onClick={() => setSelectedCert(null)}>Done</Button>
            </div>
          </div>
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
      {/* Chat Tab View */}
      {activeTab === 'Chat' && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: '500px' }}>
          <Chat />
        </div>
      )}

      {activeTab !== 'Storage' && activeTab !== 'Chat' && (
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
          <div>
            {activeTab === 'Recommended' && (
              <div style={{
                background: 'linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(217,119,6,0.06) 100%)',
                border: '1px solid var(--accent)',
                borderRadius: '8px',
                padding: '12px 16px',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <div>
                  <h3 style={{ margin: 0, color: '#fbbf24', fontSize: '15px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    🪷 Saarthi Officially Certified Base Models
                  </h3>
                  <p style={{ margin: '2px 0 0 0', fontSize: '12px', color: 'var(--text-secondary)' }}>
                    Production-tested base models certified for accuracy, stability, zero reasoning token leakage, and dynamic LoRA adapter switching.
                  </p>
                </div>
                <Badge variant="success">⭐⭐⭐⭐⭐ Officially Recommended</Badge>
              </div>
            )}

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
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
                      {model.certification ? (
                        <span style={{
                          background: 'linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.3) 100%)',
                          border: '1px solid var(--accent)',
                          color: '#fbbf24',
                          padding: '3px 8px',
                          borderRadius: '6px',
                          fontWeight: 'bold',
                          fontSize: '11px',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}>
                          ⭐⭐⭐⭐⭐ Saarthi Certified ({model.certification.confidenceScore}/100)
                        </span>
                      ) : (
                        <Badge variant={model.confidence === 'High' ? 'success' : model.confidence === 'Medium' ? 'warning' : 'default'}>
                          {model.confidence} Confidence
                        </Badge>
                      )}
                      <span className={styles.archBadge}>{model.architecture}</span>
                    </div>
                  </div>

                  {/* Why Recommended / Certification Rationale Callout */}
                  {model.certification && (
                    <div style={{
                      background: 'rgba(59,130,246,0.08)',
                      borderLeft: '3px solid var(--accent)',
                      padding: '8px 12px',
                      borderRadius: '4px',
                      marginBottom: '12px',
                      fontSize: '12px',
                      color: 'var(--text-primary)'
                    }}>
                      <div style={{ fontWeight: '600', color: 'var(--accent)', marginBottom: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Sparkles size={13} /> Why Saarthi Recommends This Base Model:
                      </div>
                      <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        {model.certification.quirksAndNotes || 'Verified instruction following, stable outputs, zero reasoning tag leakage, correct chat template & stop tokens, production-tested runtime.'}
                      </p>

                      {/* Supported LoRA Capabilities Chips */}
                      {model.certification.loraCapabilityMatrix && (
                        <div style={{ marginTop: '8px', display: 'flex', gap: '4px', flexWrap: 'wrap', alignItems: 'center' }}>
                          <span style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--text-tertiary)' }}>LoRA Ecosystem:</span>
                          {Object.entries(model.certification.loraCapabilityMatrix).map(([cap, tier]) => (
                            <span key={cap} style={{
                              fontSize: '10px',
                              padding: '2px 6px',
                              borderRadius: '4px',
                              background: tier === 'Certified' ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)',
                              border: `1px solid ${tier === 'Certified' ? 'var(--success)' : 'var(--warning)'}`,
                              color: tier === 'Certified' ? 'var(--success)' : 'var(--warning)',
                              textTransform: 'capitalize'
                            }}>
                              ⚡ {cap.replace('_', ' ')}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

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
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', width: '100%' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                          {loadedModelInfo && isSameModelId(loadedModelInfo.modelId, model.modelId) ? (
                            <Badge variant="success">● Loaded & Ready</Badge>
                          ) : (
                            <Badge variant="success">✓ Installed</Badge>
                          )}
                          <Button variant="ghost" size="sm" onClick={() => setActiveTab('Storage')}>
                            View in Storage
                          </Button>
                        </div>
                        <div style={{ display: 'flex', gap: '8px', width: '100%' }}>
                          {loadedModelInfo && isSameModelId(loadedModelInfo.modelId, model.modelId) ? (
                            <>
                              <Button variant="primary" style={{ flex: 1 }} onClick={() => navigate('/chat')}>
                                <MessageSquare size={16} style={{ marginRight: 6 }} /> Open Chat
                              </Button>
                              <Button variant="secondary" onClick={handleUnloadActiveModel}>
                                <Power size={16} style={{ marginRight: 6 }} /> Unload
                              </Button>
                            </>
                          ) : loadingModelId === model.modelId ? (
                            <Button variant="secondary" style={{ flex: 1 }} disabled>
                              <Spinner size="sm" /> Loading Model...
                            </Button>
                          ) : (
                            <Button
                              variant="primary"
                              style={{ flex: 1 }}
                              onClick={() => {
                                const im = getInstalledModel(model.modelId, model.quantization);
                                if (im) handleLoadModel(im.providerId, im.modelId, im.quantization);
                              }}
                            >
                              <PlayCircle size={16} style={{ marginRight: 6 }} /> Load Model
                            </Button>
                          )}
                        </div>
                      </div>
                    ) : downloadTask && downloadTask.status !== 'Cancelled' ? (
                      <div className={styles.progressContainer}>
                        {/* Base Model Section */}
                        <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--text-primary)', marginBottom: 4 }}>
                          Base Model ({model.modelName} - {model.quantization})
                        </div>
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

                        {/* Capability Adapters Section */}
                        <div style={{ marginTop: 12, paddingTop: 8, borderTop: '1px solid var(--border)' }}>
                          <div style={{ fontWeight: 600, fontSize: '11px', color: 'var(--accent)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Layers size={12} /> Capability LoRA Adapters
                          </div>
                          
                          {[
                            { key: 'coding', label: 'Coding' },
                            { key: 'reasoning', label: 'Reasoning' },
                            { key: 'tool-calling', label: 'Tool Calling' },
                            { key: 'mathematics', label: 'Mathematics' },
                            { key: 'research', label: 'Research' },
                          ].map((cap) => {
                            const adapterTaskId = `dl_${model.modelId.replace(/\//g, '_')}_${model.quantization.toLowerCase()}_adapter_${cap.key}`;
                            const adapterTask = activeTasks[adapterTaskId];

                            // === ADAPTER STATE MACHINE ===
                            // Priority 1: Backend manifest (single source of truth for persisted state)
                            const im = getInstalledModel(model.modelId, model.quantization);
                            const manifestAdapter = im?.adapters?.[cap.key];
                            const isInstalledInManifest = (manifestAdapter?.status === 'Installed' || manifestAdapter?.status === 'READY') && (Boolean(manifestAdapter?.adapterFile) || Boolean(manifestAdapter?.localPath));
                            const isUnavailableInManifest = manifestAdapter?.status === 'Unavailable';
                            const isFailedInManifest = manifestAdapter?.status === 'Failed';

                            // Priority 2: Active download task (transient state)
                            const taskIsUnavailable = adapterTask?.status === 'Completed' && Boolean(adapterTask.error?.includes('Unavailable'));
                            const taskIsCompleted = adapterTask?.status === 'Completed' && !taskIsUnavailable;
                            const taskIsDownloading = adapterTask?.status === 'Downloading';
                            const taskIsVerifying = adapterTask?.status === 'Verifying';
                            const taskIsSearching = adapterTask?.status === 'Resolving' || adapterTask?.speedFormatted === 'Searching...';
                            const taskIsFailed = adapterTask?.status === 'Failed';

                            // Derive final state (manifest takes priority over stale tasks)
                            let adapterState: 'Installed' | 'Unavailable' | 'Searching' | 'Downloading' | 'Verifying' | 'Failed' | 'Unknown';
                            if (isInstalledInManifest || taskIsCompleted) {
                              adapterState = 'Installed';
                            } else if (taskIsVerifying) {
                              adapterState = 'Verifying';
                            } else if (taskIsDownloading) {
                              adapterState = 'Downloading';
                            } else if (taskIsSearching) {
                              adapterState = 'Searching';
                            } else if (taskIsUnavailable || isUnavailableInManifest) {
                              adapterState = 'Unavailable';
                            } else if (taskIsFailed || isFailedInManifest) {
                              adapterState = 'Failed';
                            } else {
                              // No active task AND no manifest record → unknown/pending
                              adapterState = 'Unknown';
                            }

                            const badgeVariant = adapterState === 'Installed' ? 'success'
                              : adapterState === 'Failed' ? 'error'
                              : adapterState === 'Unavailable' ? 'default'
                              : 'warning';

                            const badgeText = adapterState === 'Installed' ? '✓ Ready'
                              : adapterState === 'Unavailable' ? 'Base Native'
                              : adapterState === 'Searching' ? 'Searching...'
                              : adapterState === 'Downloading' ? 'Downloading'
                              : adapterState === 'Verifying' ? 'Verifying...'
                              : adapterState === 'Failed' ? 'Failed'
                              : 'Base Native';

                            const showProgress = adapterState === 'Downloading' || adapterState === 'Verifying' || adapterState === 'Installed';
                            const pct = adapterTask?.progressPercent ?? (adapterState === 'Installed' ? 100 : 0);

                            return (
                              <div key={cap.key} style={{ marginBottom: 6, fontSize: '11px', background: 'var(--surface-subtle)', padding: '6px 8px', borderRadius: '6px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                                  <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                    ⚡ {cap.label}
                                  </span>
                                  <Badge variant={badgeVariant}>
                                    {badgeText}
                                  </Badge>
                                </div>
                                {showProgress && adapterTask ? (
                                  <>
                                    <div className={styles.progressBarBg} style={{ height: '4px', marginTop: 4 }}>
                                      <div className={styles.progressBarFill} style={{ width: `${Math.min(100, pct)}%` }} />
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-tertiary)', marginTop: 2 }}>
                                      <span>
                                        {adapterTask.totalBytes > 0
                                          ? `${formatBytes(adapterTask.downloadedBytes)} / ${formatBytes(adapterTask.totalBytes)}`
                                          : 'Resolving weight size...'}
                                      </span>
                                      <span>{pct.toFixed(0)}%</span>
                                    </div>
                                  </>
                                ) : adapterState === 'Installed' && !adapterTask ? (
                                  <>
                                    <div className={styles.progressBarBg} style={{ height: '4px', marginTop: 4 }}>
                                      <div className={styles.progressBarFill} style={{ width: '100%' }} />
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-tertiary)', marginTop: 2 }}>
                                      <span>{manifestAdapter?.repoId || 'LoRA Adapter'}</span>
                                      <span>100%</span>
                                    </div>
                                  </>
                                ) : null}
                              </div>
                            );
                          })}
                        </div>

                        {downloadTask.error && (
                          <div style={{ color: 'var(--error)', fontSize: '11px', marginTop: 4 }}>
                            {downloadTask.error}
                          </div>
                        )}
                        <div style={{ display: 'flex', gap: '8px', marginTop: 8 }}>
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
        </div>
        )
      )}
    </div>
  );
};