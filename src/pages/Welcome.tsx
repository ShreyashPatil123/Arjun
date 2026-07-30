import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Compass, Cpu, HardDrive, Monitor, CheckCircle2, AlertTriangle, ArrowRight, RefreshCw, Edit3, ShieldAlert, CpuIcon } from 'lucide-react';
import { Button, Card, Input, Toggle } from '../components/ui';
import { getSarathiClient } from '../sdk';
import { useToast } from '../hooks/useToast';
import styles from './Welcome.module.css';

type OnboardingStep = 'welcome' | 'scanning' | 'manual';

interface ChecklistStep {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'completed' | 'failed';
}

export const Welcome: React.FC = () => {
  const navigate = useNavigate();
  const { addToast } = useToast();

  const [step, setStep] = useState<OnboardingStep>('welcome');
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [currentScanStep, setCurrentScanStep] = useState<string>('Initializing analysis pipeline...');
  const [scanError, setScanError] = useState<string | null>(null);

  const [checklist, setChecklist] = useState<ChecklistStep[]>([
    { id: 'cpu', label: 'Detecting CPU', status: 'pending' },
    { id: 'gpu', label: 'Detecting GPU', status: 'pending' },
    { id: 'memory', label: 'Detecting Memory', status: 'pending' },
    { id: 'storage', label: 'Detecting Storage', status: 'pending' },
    { id: 'os', label: 'Detecting Operating System', status: 'pending' },
    { id: 'software', label: 'Detecting Installed Software', status: 'pending' },
    { id: 'profile', label: 'Building Hardware Profile', status: 'pending' },
  ]);

  // Manual Form State
  const [manualCpu, setManualCpu] = useState('');
  const [manualRam, setManualRam] = useState('');
  const [manualStorage, setManualStorage] = useState('');
  const [manualOs, setManualOs] = useState('');
  const [hasIntegratedGpu, setHasIntegratedGpu] = useState(true);
  const [manualGpu1, setManualGpu1] = useState('');
  const [hasDedicatedGpu, setHasDedicatedGpu] = useState(true);
  const [manualGpu2, setManualGpu2] = useState('');

  // Start Automated Scanning Flow
  const startSystemScan = async () => {
    setStep('scanning');
    setScanProgress(5);
    setScanError(null);
    setCurrentScanStep('Starting hardware detection pipeline...');

    // Reset Checklist
    setChecklist(prev => prev.map(item => ({ ...item, status: 'pending' })));

    try {
      // Step-by-step progress simulation & trigger backend execution
      const updateStep = (id: string, pct: number, label: string) => {
        setScanProgress(pct);
        setCurrentScanStep(label);
        setChecklist(prev =>
          prev.map(item => {
            if (item.id === id) return { ...item, status: 'active' };
            const ids = ['cpu', 'gpu', 'memory', 'storage', 'os', 'software', 'profile'];
            if (ids.indexOf(item.id) < ids.indexOf(id)) return { ...item, status: 'completed' };
            return item;
          })
        );
      };

      updateStep('cpu', 15, 'Detecting CPU topology & SIMD instructions...');
      await new Promise(r => setTimeout(r, 200));

      updateStep('gpu', 35, 'Interrogating GPU VRAM, Drivers, & CUDA capabilities...');
      await new Promise(r => setTimeout(r, 250));

      updateStep('memory', 50, 'Measuring system RAM & memory bus frequency...');
      await new Promise(r => setTimeout(r, 200));

      updateStep('storage', 68, 'Checking storage drives & AI model readiness...');
      await new Promise(r => setTimeout(r, 200));

      updateStep('os', 82, 'Verifying OS kernel build & platform environment...');
      await new Promise(r => setTimeout(r, 200));

      updateStep('software', 92, 'Detecting Python, Rust, Git, Node, Ollama & CUDA toolkits...');
      await new Promise(r => setTimeout(r, 250));

      updateStep('profile', 98, 'Normalizing metrics & compiling AI Capability Profile...');
      const profile = await getSarathiClient().systemAnalyzer.analyzeSystem();

      setChecklist(prev => prev.map(item => ({ ...item, status: 'completed' })));
      setScanProgress(100);
      setCurrentScanStep('System Analysis Complete!');

      addToast('success', 'System analysis complete! Navigating to specs...', 4000);

      setTimeout(() => {
        navigate('/system');
      }, 600);

    } catch (err: any) {
      console.error('System analysis failed:', err);
      setScanError(err.message || 'System analysis encountered an unexpected hardware detection issue.');
      addToast('error', 'Hardware analysis failed. You can retry or enter specs manually.', 5000);
    }
  };

  // Submit Manual Form
  const handleSaveManualInput = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const client = getSarathiClient();
      let profile = await client.systemAnalyzer.getHardwareProfile();

      if (!profile) {
        profile = await client.systemAnalyzer.analyzeSystem();
      }

      if (manualCpu) await client.systemAnalyzer.overrideHardwareValue('cpu.model', manualCpu);
      if (manualRam) {
        const ramGb = parseFloat(manualRam);
        if (!isNaN(ramGb)) {
          await client.systemAnalyzer.overrideHardwareValue('memory.totalBytes', ramGb * 1024 * 1024 * 1024);
        }
      }
      if (manualOs) await client.systemAnalyzer.overrideHardwareValue('os.edition', manualOs);

      if (hasIntegratedGpu && manualGpu1) {
        await client.systemAnalyzer.overrideHardwareValue('gpus.0.model', manualGpu1);
      }
      if (hasDedicatedGpu && manualGpu2) {
        await client.systemAnalyzer.overrideHardwareValue('gpus.1.model', manualGpu2);
      }

      addToast('success', 'Manual specifications recorded successfully.', 4000);
      navigate('/system');
    } catch (err: any) {
      addToast('error', err.message || 'Could not save manual specs.', 5000);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.orb} />

      {step === 'welcome' && (
        <div className={styles.welcomeCard}>
          <div className={styles.logoWrapper}>
            <Compass size={64} strokeWidth={1.8} className={styles.logoIcon} />
          </div>
          <h1 className={styles.title}>Welcome to Sarathi</h1>
          <p className={styles.subtitle}>A Local-First LoRA Orchestration System</p>

          <div className={styles.pills}>
            <span className={styles.pill}>Local First</span>
            <span className={styles.pill}>Privacy First</span>
            <span className={styles.pill}>Modular</span>
          </div>

          <div className={styles.actionGroup}>
            <Button
              variant="primary"
              size="lg"
              onClick={startSystemScan}
              className={styles.scanButton}
            >
              <Compass size={20} style={{ marginRight: 8 }} />
              Scan My PC
            </Button>

            <Button
              variant="ghost"
              size="md"
              onClick={() => setStep('manual')}
              className={styles.manualLink}
            >
              <Edit3 size={16} style={{ marginRight: 6 }} />
              Manually Input My PC Specifications
            </Button>
          </div>
        </div>
      )}

      {step === 'scanning' && (
        <Card className={styles.scanCard}>
          <div className={styles.scanHeader}>
            <div className={styles.scanTitleGroup}>
              <RefreshCw size={24} className={styles.spinningIcon} />
              <h2>Scanning your system...</h2>
            </div>
            <span className={styles.progressPct}>{scanProgress}%</span>
          </div>

          <div className={styles.progressBarTrack}>
            <div className={styles.progressBarFill} style={{ width: `${scanProgress}%` }} />
          </div>

          <p className={styles.currentStepText}>{currentScanStep}</p>

          <div className={styles.checklist}>
            {checklist.map(item => (
              <div key={item.id} className={styles.checkItem}>
                {item.status === 'completed' && <CheckCircle2 size={18} className={styles.checkCompleted} />}
                {item.status === 'active' && <RefreshCw size={18} className={styles.checkActive} />}
                {item.status === 'pending' && <div className={styles.checkPending} />}
                {item.status === 'failed' && <AlertTriangle size={18} className={styles.checkFailed} />}
                <span className={item.status === 'completed' ? styles.labelDone : styles.labelActive}>
                  {item.label}
                </span>
              </div>
            ))}
          </div>

          {scanError && (
            <div className={styles.errorBox}>
              <ShieldAlert size={20} className={styles.errorIcon} />
              <div>
                <p className={styles.errorTitle}>Hardware Detection Failure</p>
                <p className={styles.errorMessage}>{scanError}</p>
                <div className={styles.errorActions}>
                  <Button variant="secondary" size="sm" onClick={startSystemScan}>
                    Retry Detection
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setStep('manual')}>
                    Enter Manually
                  </Button>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}

      {step === 'manual' && (
        <Card className={styles.manualCard}>
          <div className={styles.manualHeader}>
            <Edit3 size={24} className={styles.manualHeaderIcon} />
            <div>
              <h2>Manual PC Specifications</h2>
              <p>Fallback input mode. Automatic hardware detection is preferred.</p>
            </div>
          </div>

          <form onSubmit={handleSaveManualInput} className={styles.form}>
            <div className={styles.fieldGrid}>
              <Input
                label="CPU Model"
                placeholder="e.g. AMD Ryzen 7 7840HS or Intel i9-13900K"
                value={manualCpu}
                onChange={e => setManualCpu(e.target.value)}
              />

              <Input
                label="Total System RAM (GB)"
                placeholder="e.g. 16, 32, 64"
                type="number"
                value={manualRam}
                onChange={e => setManualRam(e.target.value)}
              />

              <Input
                label="Storage Readiness"
                placeholder="e.g. 1 TB NVMe SSD"
                value={manualStorage}
                onChange={e => setManualStorage(e.target.value)}
              />

              <Input
                label="Operating System"
                placeholder="e.g. Windows 11 Pro"
                value={manualOs}
                onChange={e => setManualOs(e.target.value)}
              />
            </div>

            <div className={styles.gpuSection}>
              <h3 className={styles.sectionTitle}>
                <CpuIcon size={18} style={{ marginRight: 6, display: 'inline' }} />
                GPU Configuration
              </h3>

              <div className={styles.toggleRow}>
                <div className={styles.toggleGroup}>
                  <span>Integrated GPU</span>
                  <Toggle
                    checked={hasIntegratedGpu}
                    onChange={setHasIntegratedGpu}
                  />
                </div>

                <div className={styles.toggleGroup}>
                  <span>Dedicated GPU</span>
                  <Toggle
                    checked={hasDedicatedGpu}
                    onChange={setHasDedicatedGpu}
                  />
                </div>
              </div>

              <div className={styles.gpuFields}>
                {hasIntegratedGpu && (
                  <Input
                    label="GPU 1 (Integrated)"
                    placeholder="e.g. AMD Radeon 780M or Intel Iris Xe"
                    value={manualGpu1}
                    onChange={e => setManualGpu1(e.target.value)}
                  />
                )}

                {hasDedicatedGpu && (
                  <Input
                    label="GPU 2 (Dedicated)"
                    placeholder="e.g. NVIDIA GeForce RTX 4090 (24GB)"
                    value={manualGpu2}
                    onChange={e => setManualGpu2(e.target.value)}
                  />
                )}
              </div>
            </div>

            <div className={styles.formFooter}>
              <Button type="button" variant="ghost" onClick={() => setStep('welcome')}>
                Back
              </Button>

              <Button type="submit" variant="primary">
                Save & Continue
                <ArrowRight size={16} style={{ marginLeft: 6 }} />
              </Button>
            </div>
          </form>
        </Card>
      )}
    </div>
  );
};