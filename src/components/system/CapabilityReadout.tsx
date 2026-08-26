import React from 'react';
import { Boxes, Eye, Layers, Search } from 'lucide-react';
import type { AICapabilityProfile } from '../../types/system';
import styles from './CapabilityReadout.module.css';

interface CapabilityReadoutProps {
  capabilities: AICapabilityProfile;
  /** False before a scan has produced usable hardware data. */
  detected: boolean;
}

const formatModelSize = (bytes?: number | null) => {
  if (!bytes || bytes <= 0) return null;
  const gb = bytes / 1024 ** 3;
  return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
};

const formatContext = (tokens?: number | null) => {
  if (!tokens || tokens <= 0) return null;
  return tokens >= 1000 ? `${Math.round(tokens / 1000)}K` : String(tokens);
};

/** A capability the workbench needs, and whether this machine can supply it. */
const CapabilityFlag = ({
  icon,
  label,
  supported,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  supported: boolean;
  detail: string;
}) => (
  <div className={supported ? styles.flagOn : styles.flagOff}>
    <span className={styles.flagIcon}>{icon}</span>
    <span className={styles.flagText}>
      <span className={styles.flagLabel}>{label}</span>
      <span className={styles.flagDetail}>{detail}</span>
    </span>
  </div>
);

/**
 * What this machine can actually run.
 *
 * Replaces the old "Score: n/100" badge, which graded the hardware against
 * nothing in particular. ARJUN has one real question of a host — which models
 * fit, and which of the workbench's jobs it can therefore do — so the readout
 * answers that instead of assigning a number.
 *
 * Before a successful scan it says so rather than rendering zeroes: a panel of
 * `0 Bytes` reads as broken hardware when the truth is that nothing has looked
 * at the hardware yet.
 */
export const CapabilityReadout = ({ capabilities, detected }: CapabilityReadoutProps) => {
  if (!detected) {
    return (
      <div className={styles.undetected}>
        <span className={styles.undetectedTitle}>Hardware not yet detected</span>
        <span className={styles.undetectedDetail}>
          Run a scan to see which models this machine can hold, and which of the workbench&rsquo;s
          jobs it can run locally.
        </span>
      </div>
    );
  }

  const modelSize = formatModelSize(capabilities.maxRecommendedModelSizeBytes);
  const context = formatContext(capabilities.recommendedContextLength);
  const quant = capabilities.recommendedQuantizations?.[0];

  return (
    <div className={styles.readout}>
      <div className={styles.figures}>
        <div className={styles.figure}>
          <span className={styles.figureLabel}>Largest model that fits</span>
          <span className={styles.figureValue}>{modelSize ?? '—'}</span>
          <span className={styles.figureNote}>{quant ? `at ${quant}` : 'quantisation unknown'}</span>
        </div>
        <div className={styles.figure}>
          <span className={styles.figureLabel}>Context</span>
          <span className={styles.figureValue}>{context ?? '—'}</span>
          <span className={styles.figureNote}>tokens</span>
        </div>
        <div className={styles.figure}>
          <span className={styles.figureLabel}>Backend</span>
          <span className={styles.figureValue}>{capabilities.preferredInferenceBackend ?? '—'}</span>
          <span className={styles.figureNote}>selected for this host</span>
        </div>
      </div>

      <div className={styles.flags}>
        <CapabilityFlag
          icon={<Boxes size={15} />}
          label="Several models at once"
          supported={capabilities.multiModelCapable}
          detail={
            capabilities.multiModelCapable
              ? 'Enough memory to hold more than one'
              : 'One at a time — swapped on demand'
          }
        />
        <CapabilityFlag
          icon={<Eye size={15} />}
          label="Scans and drawings"
          supported={capabilities.visionReady}
          detail={capabilities.visionReady ? 'Vision model can run here' : 'Text only on this host'}
        />
        <CapabilityFlag
          icon={<Search size={15} />}
          label="Document search"
          supported={capabilities.embeddingReady}
          detail={
            capabilities.embeddingReady ? 'Embeddings can run here' : 'Keyword search only'
          }
        />
        <CapabilityFlag
          icon={<Layers size={15} />}
          label="Capability adapters"
          supported={capabilities.loraReady}
          detail={capabilities.loraReady ? 'LoRA adapters can be bound' : 'Prompt profiles instead'}
        />
      </div>
    </div>
  );
};
