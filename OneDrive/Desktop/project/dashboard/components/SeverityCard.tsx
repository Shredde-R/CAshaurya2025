import styles from './SeverityCard.module.css';
import type { Prediction } from '@/lib/api';

interface Props { prediction: Prediction; }

const SEVERITY_EMOJI: Record<string, string> = {
  safe: '🟢',
  medium: '🟡',
  severe: '🔴',
};

export default function SeverityCard({ prediction }: Props) {
  const sev = prediction.severity.toLowerCase() as 'safe' | 'medium' | 'severe';

  return (
    <div className={`card ${styles.card}`}>
      {/* Background glow orb */}
      <div className={`${styles.glowOrb} ${styles[`glow_${sev}`]}`} />

      <div className={styles.header}>
        <span className="label">Predicted Tide Level</span>
        <span className={`badge ${sev}`}>
          <span className={`${styles.dot} ${styles[`dot_${sev}`]}`} />
          {prediction.severity}
        </span>
      </div>

      <div className={styles.levelRow}>
        <span className={`${styles.level} ${styles[sev]}`}>
          {prediction.predictedTideLevel.toFixed(3)}
        </span>
        <span className={styles.unit}>{prediction.unit}</span>
        <span className={styles.emoji}>{SEVERITY_EMOJI[sev] ?? ''}</span>
      </div>

      <p className={styles.summary}>{prediction.summary}</p>

      <div className={styles.divider} />

      <div className={styles.observed}>
        <div className={styles.observedItem}>
          <span className="label">Observed Now</span>
          <span className={`${styles.observedValue} mono`}>
            {prediction.observedWaterLevel.toFixed(3)} {prediction.unit}
          </span>
        </div>
        <div className={styles.observedItem}>
          <span className="label">Classifier</span>
          <span className={`badge ${prediction.classifierSeverity.toLowerCase()}`}>
            {prediction.classifierSeverity}
          </span>
        </div>
        <div className={styles.observedItem}>
          <span className="label">As of</span>
          <span className={styles.observedValue}>{prediction.lastUpdated}</span>
        </div>
      </div>
    </div>
  );
}
