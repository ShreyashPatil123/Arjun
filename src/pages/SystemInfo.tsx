import React, { useEffect, useState, useCallback } from 'react';
import {
  Cpu,
  Zap,
  Database,
  HardDrive,
  Terminal,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Sliders,
  Check,
  X,
  Info,
  Server,
  Layers,
  Activity,
  ShieldCheck,
  Box
} from 'lucide-react';
import { Card, Button, Badge, Dialog, Input, Spinner } from '../components/ui';
import { useToast } from '../hooks/useToast';
import * as systemService from '../services/system.service';
import { HardwareProfile, StorageInfo, SoftwareDetectorInfo } from '../types/system';
import {
  formatBytes,
  formatFrequency,
  formatPercentage,
  formatNumber,
  classNames
} from '../utils/helpers';
import styles from './SystemInfo.module.css';

export const SystemInfo: React.FC = () => {
  const { addToast } = useToast();
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // Manual Override Dialog State
  const [overrideModalOpen, setOverrideModalOpen] = useState<boolean>(false);
  const [overrideField, setOverrideField] = useState<string>('');
  const [overrideLabel, setOverrideLabel] = useState<string>('');
  const [detectedValue, setDetectedValue] = useState<string>('');
  const [overrideInput, setOverrideInput] = useState<string>('');
  const [isSubmittingOverride, setIsSubmittingOverride] = useState<boolean>(false);

  const fetchProfile = useCallback(async () => {
    try {
      setLoading(true);
      const data = await systemService.getHardwareProfile();
      setProfile(data);
    } catch (err) {
      console.error('Failed to load hardware profile:', err);
      addToast('error', 'Failed to load hardware profile');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const handleReanalyze = async () => {
    try {
      setIsAnalyzing(true);
      const newProfile = await systemService.analyzeSystem();
      setProfile(newProfile);
      addToast('success', 'Hardware re-analysis complete');
    } catch (err) {
      console.error('Re-analyze failed:', err);
      addToast('error', 'Failed to re-analyze hardware');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const openOverrideModal = (fieldPath: string, label: string, currentVal: unknown) => {
    setOverrideField(fieldPath);
    setOverrideLabel(label);
    const valStr = currentVal !== null && currentVal !== undefined ? String(currentVal) : '';
    setDetectedValue(valStr);
    
    // Check if field is currently overridden in profile.overrides
    const existingOverride = profile?.overrides?.[fieldPath];
    setOverrideInput(existingOverride !== undefined && existingOverride !== null ? String(existingOverride) : valStr);
    setOverrideModalOpen(true);
  };

  const handleSaveOverride = async () => {
    if (!overrideField) return;
    try {
      setIsSubmittingOverride(true);
      let parsedValue: unknown = overrideInput.trim();
      if (!isNaN(Number(parsedValue)) && parsedValue !== '') {
        parsedValue = Number(parsedValue);
      } else if (parsedValue === 'true') {
        parsedValue = true;
      } else if (parsedValue === 'false') {
        parsedValue = false;
      }

      const updatedProfile = await systemService.overrideHardwareValue(overrideField, parsedValue);
      setProfile(updatedProfile);
      addToast('success', `Updated override for ${overrideLabel}`);
      setOverrideModalOpen(false);
    } catch (err) {
      console.error('Save override error:', err);
      addToast('error', 'Failed to apply override');
    } finally {
      setIsSubmittingOverride(false);
    }
  };

  const handleRevertOverride = async () => {
    if (!overrideField) return;
    try {
      setIsSubmittingOverride(true);
      const updatedProfile = await systemService.revertHardwareOverride(overrideField);
      setProfile(updatedProfile);
      addToast('info', `Reverted override for ${overrideLabel}`);
      setOverrideModalOpen(false);
    } catch (err) {
      console.error('Revert override error:', err);
      addToast('error', 'Failed to revert override');
    } finally {
      setIsSubmittingOverride(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', flexDirection: 'column', gap: '16px' }}>
        <Spinner size="lg" color="var(--accent)" />
        <span style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Analyzing Hardware & System Architecture...</span>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className={styles.container}>
        <Card>
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <AlertTriangle size={36} color="var(--warning)" style={{ marginBottom: '12px' }} />
            <p>No hardware profile detected.</p>
            <Button onClick={fetchProfile} style={{ marginTop: '16px' }}>Retry Scan</Button>
          </div>
        </Card>
      </div>
    );
  }

  const { cpu, gpus, memory, storage, os, software, aiRuntimes, aiCapabilities, validation, overrides } = profile;

  // Percentage calculations via helpers (zero inline calculations)
  const memoryUsedPercentage = formatPercentage(memory.usedBytes, memory.totalBytes);
  const primaryGpu = gpus[0];
  const gpuVramUsedBytes = primaryGpu ? primaryGpu.vramTotalBytes - primaryGpu.vramFreeBytes : 0;
  const gpuVramPercentage = primaryGpu ? formatPercentage(gpuVramUsedBytes, primaryGpu.vramTotalBytes) : 0;

  const isFieldOverridden = (fieldPath: string) => Boolean(overrides && overrides[fieldPath] !== undefined);

  return (
    <div className={styles.container}>
      {/* Header Section */}
      <div className={styles.header}>
        <div className={styles.headerInfo}>
          <div className={styles.titleRow}>
            <h1 className={styles.headerTitle}>System Analyzer</h1>
            <Badge variant={validation.isReadyForAi ? 'success' : 'warning'} dot={false}>
              Score: {validation.score}/100 &bull; {validation.isReadyForAi ? 'AI Ready' : 'Optimization Required'}
            </Badge>
          </div>
          <p className={styles.headerSubtitle}>
            Hardware Architecture &bull; GPU Acceleration &bull; AI Runtime Compatibility Assessment
          </p>
        </div>

        <div className={styles.headerActions}>
          <Button
            variant="secondary"
            icon={<RotateCcw size={16} className={isAnalyzing ? styles.spinningIcon : ''} />}
            onClick={handleReanalyze}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? 'Scanning Hardware...' : 'Re-analyze System'}
          </Button>
        </div>
      </div>

      {/* Main Grid */}
      <div className={styles.grid}>
        {/* CPU Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Cpu size={20} /></div>
                <span className={styles.cardTitle}>Processor (CPU)</span>
              </div>
              <button
                className={styles.overrideTriggerBtn}
                onClick={() => openOverrideModal('cpu.model', 'CPU Model', cpu.model)}
                title="Override CPU Spec"
              >
                <Sliders size={13} />
                <span>Override</span>
              </button>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Model Name</span>
              <span className={styles.specValue}>
                {cpu.model}
                {isFieldOverridden('cpu.model') && <span className={styles.overrideBadge}>Overridden</span>}
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Architecture</span>
              <span className={styles.specValue}>{cpu.architecture} ({cpu.manufacturer})</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Cores / Threads</span>
              <span className={styles.specValue}>{cpu.physicalCores} Cores / {cpu.logicalProcessors} Threads</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Base / Boost Frequency</span>
              <span className={styles.specValue}>
                {formatFrequency(cpu.baseFrequencyMhz)} / {formatFrequency(cpu.boostFrequencyMhz)}
              </span>
            </div>
            {cpu.cacheL3Kb && (
              <div className={styles.specRow}>
                <span className={styles.specLabel}>L3 Cache Size</span>
                <span className={styles.specValue}>{formatBytes(cpu.cacheL3Kb * 1024)}</span>
              </div>
            )}
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Hardware Virtualization</span>
              <span className={styles.specValue}>
                {cpu.virtualizationSupported ? (
                  <Badge variant="success">Supported</Badge>
                ) : (
                  <Badge variant="warning">Disabled / Unsupported</Badge>
                )}
              </span>
            </div>

            <div style={{ marginTop: '8px' }}>
              <span className={styles.specLabel} style={{ marginBottom: '6px' }}>Instruction Sets & SIMD:</span>
              <div className={styles.pillsGroup}>
                {cpu.simdCapabilities.map(cap => (
                  <span key={cap} className={styles.badgeTag}>{cap}</span>
                ))}
              </div>
            </div>
          </div>
        </Card>

        {/* GPU Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Zap size={20} /></div>
                <span className={styles.cardTitle}>Graphics & AI Acceleration (GPU)</span>
              </div>
              {primaryGpu && (
                <button
                  className={styles.overrideTriggerBtn}
                  onClick={() => openOverrideModal('gpus.0.vramTotalBytes', 'GPU VRAM Total Bytes', primaryGpu.vramTotalBytes)}
                  title="Override GPU VRAM Spec"
                >
                  <Sliders size={13} />
                  <span>Override VRAM</span>
                </button>
              )}
            </div>
          }
        >
          {gpus.length === 0 ? (
            <div style={{ padding: '16px', color: 'var(--text-secondary)', fontSize: '13px' }}>
              No dedicated graphics accelerators detected. CPU-only inference backend will be active.
            </div>
          ) : (
            gpus.map((gpu, idx) => {
              const gpuVramAllocated = gpu.vramTotalBytes - gpu.vramFreeBytes;
              const gpuPct = formatPercentage(gpuVramAllocated, gpu.vramTotalBytes);
              return (
                <div key={idx} className={styles.specList}>
                  <div className={styles.specRow}>
                    <span className={styles.specLabel}>Device Model</span>
                    <span className={styles.specValue}>
                      {gpu.model} ({gpu.vendor})
                      {gpu.isDedicated && <Badge variant="info">Dedicated</Badge>}
                    </span>
                  </div>

                  <div className={styles.progressSection}>
                    <div className={styles.progressHeader}>
                      <span>VRAM Allocation</span>
                      <span>
                        {formatBytes(gpuVramAllocated)} used of {formatBytes(gpu.vramTotalBytes)} ({gpuPct}%)
                      </span>
                    </div>
                    <div className={styles.progressBarTrack}>
                      <div className={styles.progressBarFill} style={{ width: `${gpuPct}%` }} />
                    </div>
                  </div>

                  <div className={styles.specRow}>
                    <span className={styles.specLabel}>Driver Version</span>
                    <span className={styles.specValue}>{gpu.driverVersion || 'N/A'}</span>
                  </div>
                  {gpu.computeCapability && (
                    <div className={styles.specRow}>
                      <span className={styles.specLabel}>Compute Capability</span>
                      <span className={styles.specValue}>v{gpu.computeCapability}</span>
                    </div>
                  )}

                  <div style={{ marginTop: '8px' }}>
                    <span className={styles.specLabel} style={{ marginBottom: '6px' }}>Acceleration API Support:</span>
                    <div className={styles.pillsGroup}>
                      <Badge variant={gpu.cudaSupported ? 'success' : 'default'}>
                        CUDA: {gpu.cudaSupported ? 'Available' : 'No'}
                      </Badge>
                      <Badge variant={gpu.vulkanSupported ? 'success' : 'default'}>
                        Vulkan: {gpu.vulkanSupported ? 'Available' : 'No'}
                      </Badge>
                      <Badge variant={gpu.directxSupported ? 'success' : 'default'}>
                        DirectX: {gpu.directxSupported ? 'Available' : 'No'}
                      </Badge>
                      <Badge variant={gpu.openclSupported ? 'info' : 'default'}>
                        OpenCL: {gpu.openclSupported ? 'Available' : 'No'}
                      </Badge>
                      <Badge variant={gpu.rocmSupported ? 'success' : 'default'}>
                        ROCm: {gpu.rocmSupported ? 'Available' : 'No'}
                      </Badge>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </Card>

        {/* System Memory Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Database size={20} /></div>
                <span className={styles.cardTitle}>System Memory (RAM)</span>
              </div>
              <button
                className={styles.overrideTriggerBtn}
                onClick={() => openOverrideModal('memory.totalBytes', 'System RAM Total Bytes', memory.totalBytes)}
                title="Override RAM Spec"
              >
                <Sliders size={13} />
                <span>Override</span>
              </button>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.progressSection}>
              <div className={styles.progressHeader}>
                <span>RAM Usage</span>
                <span>
                  {formatBytes(memory.usedBytes)} used of {formatBytes(memory.totalBytes)} ({memoryUsedPercentage}%)
                  {isFieldOverridden('memory.totalBytes') && <span className={styles.overrideBadge} style={{ marginLeft: '6px' }}>Overridden</span>}
                </span>
              </div>
              <div className={styles.progressBarTrack}>
                <div className={styles.progressBarFill} style={{ width: `${memoryUsedPercentage}%` }} />
              </div>
            </div>

            <div className={styles.specRow}>
              <span className={styles.specLabel}>Available Memory</span>
              <span className={styles.specValue}>{formatBytes(memory.availableBytes)}</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Memory Type & Speed</span>
              <span className={styles.specValue}>
                {memory.memoryType} @ {formatNumber(memory.speedMts)} MT/s
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Memory Slots</span>
              <span className={styles.specValue}>
                {memory.populatedSlots} / {memory.totalSlots} Slots Populated
              </span>
            </div>
          </div>
        </Card>

        {/* Storage Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><HardDrive size={20} /></div>
                <span className={styles.cardTitle}>Storage & Model Drives</span>
              </div>
            </div>
          }
        >
          <div>
            {storage.map((drive: StorageInfo, idx: number) => {
              const driveUsed = drive.totalBytes - drive.freeBytes;
              const drivePct = formatPercentage(driveUsed, drive.totalBytes);
              return (
                <div key={idx} className={styles.driveCard}>
                  <div className={styles.driveHeader}>
                    <div className={styles.driveTitle}>
                      <HardDrive size={15} color="var(--accent)" />
                      <span>{drive.driveName} ({drive.mountPoint})</span>
                    </div>
                    {drive.isAiStorageReady ? (
                      <Badge variant="success">AI Ready ({drive.driveType})</Badge>
                    ) : (
                      <Badge variant="warning">Standard ({drive.driveType})</Badge>
                    )}
                  </div>

                  <div className={styles.progressSection} style={{ margin: '4px 0 0 0' }}>
                    <div className={styles.progressHeader}>
                      <span>{drive.fileSystem} File System</span>
                      <span>
                        {formatBytes(drive.freeBytes)} Free of {formatBytes(drive.totalBytes)} ({drivePct}% Used)
                      </span>
                    </div>
                    <div className={styles.progressBarTrack}>
                      <div className={styles.progressBarFill} style={{ width: `${drivePct}%` }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* OS & Environment Card */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Terminal size={20} /></div>
                <span className={styles.cardTitle}>Operating System & Software Tools</span>
              </div>
              <button
                className={styles.overrideTriggerBtn}
                onClick={() => openOverrideModal('os.version', 'OS Version', os.version)}
                title="Override OS Version"
              >
                <Sliders size={13} />
                <span>Override</span>
              </button>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Operating System</span>
              <span className={styles.specValue}>
                {os.name} {os.edition} ({os.architecture})
                {isFieldOverridden('os.version') && <span className={styles.overrideBadge}>Overridden</span>}
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Version & Build</span>
              <span className={styles.specValue}>v{os.version} (Build {os.buildNumber})</span>
            </div>

            <div style={{ marginTop: '10px' }}>
              <span className={styles.specLabel}>Installed Developer Tools & Environment:</span>
              <div className={styles.softwareGrid}>
                {Object.entries(software).map(([key, item]) => {
                  const detector = item as SoftwareDetectorInfo;
                  if (!detector || typeof detector !== 'object') return null;
                  return (
                    <div key={key} className={styles.softwareItem}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {detector.installed ? (
                          <Check size={14} color="var(--success)" />
                        ) : (
                          <X size={14} color="var(--text-tertiary)" />
                        )}
                        <span className={styles.softwareName}>{detector.name || key}</span>
                      </div>
                      <span className={styles.softwareVersion}>
                        {detector.installed ? detector.version || 'Installed' : 'Missing'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {aiRuntimes.length > 0 && (
              <div style={{ marginTop: '14px', borderTop: '1px solid rgba(255, 255, 255, 0.04)', paddingTop: '10px' }}>
                <span className={styles.specLabel} style={{ marginBottom: '8px' }}>Detected AI Runtime Services:</span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {aiRuntimes.map((rt, i) => (
                    <div key={i} className={styles.specRow} style={{ borderBottom: 'none' }}>
                      <span className={styles.specLabel}>
                        <Server size={14} color="var(--accent)" />
                        {rt.name}
                      </span>
                      <span className={styles.specValue}>
                        <Badge variant={rt.status === 'running' ? 'success' : 'default'}>
                          {rt.status.toUpperCase()}
                        </Badge>
                        {rt.endpoint && <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{rt.endpoint}</span>}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* AI Capability Profile */}
        <Card
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><Sparkles size={20} /></div>
                <span className={styles.cardTitle}>AI Capability Profile & Recommendations</span>
              </div>
            </div>
          }
        >
          <div className={styles.specList}>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Max Recommended Model Memory</span>
              <span className={styles.specValue} style={{ color: 'var(--accent)', fontWeight: 600 }}>
                {formatBytes(aiCapabilities.maxRecommendedModelSizeBytes)}
              </span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Optimal Inference Backend</span>
              <span className={styles.specValue}>{aiCapabilities.preferredInferenceBackend}</span>
            </div>
            <div className={styles.specRow}>
              <span className={styles.specLabel}>Recommended Context Window</span>
              <span className={styles.specValue}>{formatNumber(aiCapabilities.recommendedContextLength)} Tokens</span>
            </div>

            <div className={styles.specRow}>
              <span className={styles.specLabel}>Recommended Quantizations</span>
              <div className={styles.pillsGroup}>
                {aiCapabilities.recommendedQuantizations.map(q => (
                  <span key={q} className={styles.badgeTag}>{q}</span>
                ))}
              </div>
            </div>

            <div style={{ marginTop: '8px' }}>
              <span className={styles.specLabel} style={{ marginBottom: '8px' }}>Module & Feature Readiness:</span>
              <div className={styles.pillsGroup}>
                <Badge variant={aiCapabilities.loraReady ? 'success' : 'warning'}>
                  {aiCapabilities.loraReady ? '✓ Dynamic LoRA Ready' : 'LoRA Restricted'}
                </Badge>
                <Badge variant={aiCapabilities.visionReady ? 'success' : 'default'}>
                  {aiCapabilities.visionReady ? '✓ Multimodal Vision Ready' : 'Vision Limited'}
                </Badge>
                <Badge variant={aiCapabilities.embeddingReady ? 'success' : 'default'}>
                  {aiCapabilities.embeddingReady ? '✓ RAG Vector Embeddings' : 'Embedding Limited'}
                </Badge>
                <Badge variant={aiCapabilities.multiModelCapable ? 'success' : 'default'}>
                  {aiCapabilities.multiModelCapable ? '✓ Multi-Model Parallel' : 'Single Model Only'}
                </Badge>
              </div>
            </div>
          </div>
        </Card>

        {/* System Validation & Diagnostics */}
        <Card
          className={styles.fullWidth}
          header={
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleWrapper}>
                <div className={styles.cardIcon}><ShieldCheck size={20} /></div>
                <span className={styles.cardTitle}>System Validation & Diagnostic Assessment</span>
              </div>
            </div>
          }
        >
          {validation.warnings.length > 0 && (
            <div className={classNames(styles.diagnosticBox, styles.warningBox)}>
              <div className={styles.diagTitle}>
                <AlertTriangle size={16} />
                <span>System Warnings ({validation.warnings.length})</span>
              </div>
              <ul className={styles.diagList}>
                {validation.warnings.map((warn, i) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {validation.errors.length > 0 && (
            <div className={classNames(styles.diagnosticBox, styles.errorBox)}>
              <div className={styles.diagTitle}>
                <XCircle size={16} />
                <span>System Blockers ({validation.errors.length})</span>
              </div>
              <ul className={styles.diagList}>
                {validation.errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          {validation.recommendations.length > 0 && (
            <div className={classNames(styles.diagnosticBox, styles.recBox)}>
              <div className={styles.diagTitle}>
                <Info size={16} />
                <span>System Optimization Recommendations ({validation.recommendations.length})</span>
              </div>
              <ul className={styles.diagList}>
                {validation.recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>

      {/* Manual Override Controls Dialog */}
      <Dialog
        isOpen={overrideModalOpen}
        onClose={() => setOverrideModalOpen(false)}
        title={`Manual Spec Override — ${overrideLabel}`}
        actions={
          <div className={styles.dialogActions}>
            <Button
              variant="danger"
              onClick={handleRevertOverride}
              loading={isSubmittingOverride}
              icon={<RotateCcw size={14} />}
            >
              Revert to Detected
            </Button>
            <div style={{ display: 'flex', gap: '8px' }}>
              <Button
                variant="ghost"
                onClick={() => setOverrideModalOpen(false)}
                disabled={isSubmittingOverride}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleSaveOverride}
                loading={isSubmittingOverride}
                icon={<Check size={14} />}
              >
                Save Override
              </Button>
            </div>
          </div>
        }
      >
        <div className={styles.dialogForm}>
          <div className={styles.dialogRow}>
            <label className={styles.dialogLabel}>Target Spec Field Path</label>
            <div className={styles.detectedValueBox}>{overrideField}</div>
          </div>

          <div className={styles.dialogRow}>
            <label className={styles.dialogLabel}>Original Detected Value</label>
            <div className={styles.detectedValueBox}>{detectedValue || '(Empty / Unset)'}</div>
          </div>

          <div className={styles.dialogRow}>
            <Input
              label="New Override Value"
              value={overrideInput}
              onChange={e => setOverrideInput(e.target.value)}
              placeholder="Enter custom override value"
            />
            <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
              Overrides will persist in local Sarathi configuration and adjust AI capability profiling.
            </span>
          </div>
        </div>
      </Dialog>
    </div>
  );
};