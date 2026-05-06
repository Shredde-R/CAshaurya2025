import styles from './ConfidencePanel.module.css';
import type { Confidence } from '@/lib/api';

interface Props { confidence: Confidence; }

function ProbBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className={styles.probRow}>
      <span className={styles.probLabel}>{label}</span>
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className={styles.probValue}>{pct}%</span>
    </div>
  );
}

export default function ConfidencePanel({ confidence }: Props) {
  const score = confidence.score;
  const scoreColor =
    score >= 80 ? 'var(--safe)' : score >= 60 ? 'var(--medium)' : 'var(--severe)';

  return (
    <div className={`card ${styles.card}`}>
      <span className="label">Model Confidence</span>
      <h2 className={styles.title}>{confidence.classification}</h2>

      {/* Score ring */}
      <div className={styles.scoreWrap}>
        <svg viewBox="0 0 100 100" className={styles.ring}>
          <circle cx="50" cy="50" r="42" className={styles.ringTrack} />
          <circle
            cx="50"
            cy="50"
            r="42"
            className={styles.ringFill}
            style={{
              strokeDasharray: `${2 * Math.PI * 42}`,
              strokeDashoffset: `${2 * Math.PI * 42 * (1 - score / 100)}`,
              stroke: scoreColor,
            }}
          />
        </svg>
        <div className={styles.scoreInner}>
          <span className={styles.scoreNum} style={{ color: scoreColor }}>
            {score.toFixed(1)}
          </span>
          <span className={styles.scorePct}>%</span>
          <span className={styles.scoreLabel}>{confidence.label}</span>
        </div>
      </div>

      {/* Probability bars */}
      <div className={styles.probSection}>
        <span className="label" style={{ marginBottom: 12, display: 'block' }}>
          Class Probabilities
        </span>
        <ProbBar label="Safe"   value={confidence.probabilities.safe}   color="var(--safe)" />
        <ProbBar label="Medium" value={confidence.probabilities.medium} color="var(--medium)" />
        <ProbBar label="Severe" value={confidence.probabilities.severe} color="var(--severe)" />
      </div>

      <p className={styles.note}>{confidence.note}</p>
    </div>
  );
}
