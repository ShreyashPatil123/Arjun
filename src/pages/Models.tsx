import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sparkles, AlertTriangle, CheckCircle2, ArrowLeft, RefreshCw, Layers, Zap, ShieldAlert } from 'lucide-react';
import { Card, Button, Badge, Spinner } from '../components/ui';
import { useToast } from '../hooks/useToast';
import { getModelRecommendations } from '../services/recommendation.service';
import type { ModelRecommendation, FitCategory } from '../types/recommendation';
import styles from './Models.module.css';

export const Models: React.FC = () => {
  const navigate = useNavigate();
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<FitCategory>('Recommended');
  const { addToast } = useToast();

  const loadRecommendations = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getModelRecommendations();
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

  useEffect(() => {
    loadRecommendations();
  }, [loadRecommendations]);

  const recommendedModels = recommendations.filter((r) => r.category === 'Recommended');
  const compatibleModels = recommendations.filter((r) => r.category === 'Compatible');
  const mayRunModels = recommendations.filter((r) => r.category === 'MayRun');

  const currentModels =
    activeTab === 'Recommended'
      ? recommendedModels
      : activeTab === 'Compatible'
      ? compatibleModels
      : mayRunModels;

  const formatBytes = (bytes: number) => {
    if (!bytes || bytes === 0) return '0 B';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerInfo}>
          <div className={styles.titleRow}>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/system')}
              style={{ marginRight: 8 }}
            >
              <ArrowLeft size={16} style={{ marginRight: 4 }} />
              System Specs
            </Button>
            <Sparkles size={24} color="var(--accent)" />
            <h1 className={styles.headerTitle}>AI Model Recommendations</h1>
          </div>
          <p className={styles.headerSubtitle}>
            Calculated deterministically from your PC's actual hardware specs & memory budgets
          </p>
        </div>
        <div className={styles.headerActions}>
          <Button variant="secondary" onClick={loadRecommendations} disabled={loading}>
            <RefreshCw size={14} className={loading ? styles.spinningIcon : ''} style={{ marginRight: 6 }} />
            {loading ? 'Analyzing...' : 'Refresh'}
          </Button>
        </div>
      </header>

      {/* Tabs */}
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
      </div>

      {/* Disclaimer banner for May Run tab */}
      {activeTab === 'MayRun' && (
        <div className={styles.disclaimerBanner}>
          <AlertTriangle size={18} />
          <span>
            These models may run on your system, but performance and stability are not guaranteed due to tight memory headroom or heavy offloading.
          </span>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className={styles.disclaimerBanner} style={{ borderColor: 'var(--error)', background: 'rgba(239,68,68,0.1)' }}>
          <ShieldAlert size={20} color="var(--error)" />
          <div style={{ flex: 1 }}>
            <strong style={{ color: 'var(--error)' }}>Failed to calculate recommendations</strong>
            <p style={{ margin: '4px 0 0 0', fontSize: '12px' }}>{error}</p>
          </div>
          <Button variant="secondary" size="sm" onClick={loadRecommendations}>
            Retry Calculation
          </Button>
        </div>
      )}

      {/* Cards Grid */}
      {loading ? (
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
          {currentModels.map((model) => (
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
                  <span className={styles.specLabel}>Shared Memory</span>
                  <span className={styles.specVal}>{formatBytes(model.estimatedSharedMemBytes)}</span>
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
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};