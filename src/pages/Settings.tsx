import React, { useEffect, useState } from 'react';
import { Card, Toggle, Input, Button, Badge } from '../components/ui';
import { useTheme } from '../hooks/useTheme';
import { useConfig } from '../hooks/useConfig';
import { useToast } from '../hooks/useToast';
import { getInferenceStatus } from '../services/ai.service';
import { getModelProfile, updateModelProfile, refreshModelProfile } from '../services/intelligence.service';
import type { ModelProfile, InferenceParameters } from '../types/intelligence';
import type { LoadedModelInfo } from '../types/ai';
import styles from './Settings.module.css';

export const Settings: React.FC = () => {
  const { resolvedTheme, setTheme } = useTheme();
  const { config } = useConfig();
  const { addToast } = useToast();

  const [activeModel, setActiveModel] = useState<LoadedModelInfo | null>(null);
  const [profile, setProfile] = useState<ModelProfile | null>(null);
  const [loadingProfile, setLoadingProfile] = useState<boolean>(false);
  const [params, setParams] = useState<InferenceParameters | null>(null);

  useEffect(() => {
    async function loadActiveProfile() {
      try {
        const status = await getInferenceStatus();
        if (status.model) {
          setActiveModel(status.model);
          setLoadingProfile(true);
          const prof = await getModelProfile('huggingface', status.model.modelId);
          setProfile(prof);
          setParams(prof.activeUserParams || prof.recommendedParams);
        }
      } catch (e) {
        console.log('No active model profile found:', e);
      } finally {
        setLoadingProfile(false);
      }
    }
    loadActiveProfile();
  }, []);

  const handleSaveParams = async () => {
    if (!activeModel || !params) return;
    try {
      const updated = await updateModelProfile('huggingface', activeModel.modelId, params);
      setProfile(updated);
      addToast('success', 'Model Intelligence settings updated successfully');
    } catch (e) {
      addToast('error', `Failed to update profile: ${String(e)}`);
    }
  };

  const handleResetRecommended = async () => {
    if (!profile) return;
    setParams(profile.recommendedParams);
    if (activeModel) {
      try {
        const updated = await updateModelProfile('huggingface', activeModel.modelId, profile.recommendedParams);
        setProfile(updated);
        addToast('info', 'Reset settings to recommended values');
      } catch (e) {
        addToast('error', `Failed to reset profile: ${String(e)}`);
      }
    }
  };

  const handleRefreshProfile = async () => {
    if (!activeModel) return;
    try {
      setLoadingProfile(true);
      const refreshed = await refreshModelProfile('huggingface', activeModel.modelId);
      setProfile(refreshed);
      setParams(refreshed.activeUserParams || refreshed.recommendedParams);
      addToast('success', 'Model metadata profile refreshed from local package sources');
    } catch (e) {
      addToast('error', `Failed to refresh profile: ${String(e)}`);
    } finally {
      setLoadingProfile(false);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Settings</h1>

      {/* Active Model Intelligence */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h2 className={styles.sectionTitle} style={{ margin: 0 }}>Model Intelligence Layer</h2>
            <span className={styles.desc}>Automatic local profiling & dynamic capability configuration</span>
          </div>
          {activeModel && (
            <Button variant="secondary" size="sm" onClick={handleRefreshProfile} disabled={loadingProfile}>
              Refresh Model Profile
            </Button>
          )}
        </div>

        {activeModel ? (
          profile ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className={styles.row}>
                <div className={styles.info}>
                  <span className={styles.label}>{profile.modelName}</span>
                  <span className={styles.desc}>Family: <strong>{String(profile.modelFamily).toUpperCase()}</strong> | Template: <code>{profile.chatTemplate}</code></span>
                </div>
                <Badge variant="info">Profile v{profile.profileVersion}</Badge>
              </div>

              <div>
                <span className={styles.label} style={{ fontSize: '13px', display: 'block', marginBottom: '8px' }}>Supported Capabilities</span>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {Object.entries(profile.capabilityRegistry.capabilities || {}).map(([key, item]) => (
                    <Badge key={key} variant={item.supported ? 'success' : 'default'}>
                      {key.toUpperCase()} {item.supported ? '✓' : '✗'}
                    </Badge>
                  ))}
                </div>
              </div>

              {params && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '12px' }}>
                  <div>
                    <label className={styles.label} style={{ fontSize: '12px' }}>Temperature ({params.temperature})</label>
                    <input
                      type="range"
                      min="0.0"
                      max="1.5"
                      step="0.05"
                      value={params.temperature}
                      onChange={(e) => setParams({ ...params, temperature: parseFloat(e.target.value) })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <label className={styles.label} style={{ fontSize: '12px' }}>Top-P ({params.topP})</label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={params.topP}
                      onChange={(e) => setParams({ ...params, topP: parseFloat(e.target.value) })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <Input
                    label="Max Tokens"
                    type="number"
                    value={String(params.maxTokens)}
                    onChange={(e) => setParams({ ...params, maxTokens: parseInt(e.target.value) || 2048 })}
                  />
                  <Input
                    label="Context Length"
                    type="number"
                    value={String(params.contextLength)}
                    onChange={(e) => setParams({ ...params, contextLength: parseInt(e.target.value) || 4096 })}
                  />
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '12px' }}>
                <Button variant="ghost" size="sm" onClick={handleResetRecommended}>
                  Reset to Recommended Values
                </Button>
                <Button variant="primary" size="sm" onClick={handleSaveParams}>
                  Save Runtime Parameters
                </Button>
              </div>
            </div>
          ) : (
            <div className={styles.desc}>Loading active model intelligence profile...</div>
          )
        ) : (
          <div className={styles.desc}>No active model loaded. Load a model from Storage & Models to view its intelligence profile.</div>
        )}
      </Card>

      <Card>
        <h2 className={styles.sectionTitle}>Appearance</h2>
        <div className={styles.row}>
          <div className={styles.info}>
            <span className={styles.label}>Dark Theme</span>
            <span className={styles.desc}>Toggle dark mode on or off.</span>
          </div>
          <Toggle 
            checked={resolvedTheme === 'dark'} 
            onChange={(checked) => setTheme(checked ? 'dark' : 'light')} 
          />
        </div>
      </Card>

      <Card>
        <h2 className={styles.sectionTitle}>Directories</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <Input label="Model Directory" value={config?.modelDirectory || ''} readOnly disabled />
          <Input label="Download Directory" value={config?.downloadDirectory || ''} readOnly disabled />
          <Input label="Cache Directory" value={config?.cacheDirectory || ''} readOnly disabled />
        </div>
      </Card>

      <Card>
        <h2 className={styles.sectionTitle}>About</h2>
        <div className={styles.row}>
          <div className={styles.info}>
            <span className={styles.label}>Sarathi Version</span>
            <span className={styles.desc}>v0.1.0 (Phase 5 Model Intelligence Layer)</span>
          </div>
          <Button variant="secondary">Check for Updates</Button>
        </div>
      </Card>
    </div>
  );
};