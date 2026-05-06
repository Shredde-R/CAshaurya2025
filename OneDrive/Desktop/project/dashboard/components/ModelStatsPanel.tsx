import styles from './ModelStatsPanel.module.css';
import type { ModelPerformance } from '@/lib/api';

interface Props { performance: ModelPerformance; }

function StatRow({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className={styles.statRow}>
      <span className={styles.statLabel}>{label}</span>
      <span className={`${styles.statValue} mono ${good ? styles.good : styles.neutral}`}>
        {value}
      </span>
    </div>
  );
}

export default function ModelStatsPanel({ performance }: Props) {
  const { regression: reg, classification: cls, severityFromRegression: sfr } = performance;
  return (
    <div className={`card ${styles.card}`}>
      <span className="label">Model Performance</span>
      <h2 className={styles.title}>LSTM Metrics on Test Set</h2>
      <div className={styles.sections}>
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>📈</span>
            <span className={styles.sectionName}>Tide Height Regression</span>
          </div>
          <StatRow label="MAE"  value={`${reg.mae.toFixed(4)} m`}  good={reg.mae < 0.05} />
          <StatRow label="RMSE" value={`${reg.rmse.toFixed(4)} m`} good={reg.rmse < 0.08} />
          <StatRow label="R²"   value={reg.r2.toFixed(4)}          good={reg.r2 > 0.95} />
        </div>
        <div className={styles.sectionDivider} />
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🏷</span>
            <span className={styles.sectionName}>Severity Classifier</span>
          </div>
          <StatRow label="Accuracy"    value={`${cls.accuracy.toFixed(1)}%`} good={cls.accuracy > 90} />
          <StatRow label="F1 Macro"    value={cls.f1Macro.toFixed(4)}        good={cls.f1Macro > 0.9} />
          <StatRow label="F1 Weighted" value={cls.f1Weighted.toFixed(4)}     good={cls.f1Weighted > 0.9} />
        </div>
        <div className={styles.sectionDivider} />
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionIcon}>🔀</span>
            <span className={styles.sectionName}>Severity via Regression</span>
          </div>
          <StatRow label="Accuracy"    value={`${sfr.accuracy.toFixed(1)}%`} good={sfr.accuracy > 90} />
          <StatRow label="F1 Macro"    value={sfr.f1Macro.toFixed(4)}        good={sfr.f1Macro > 0.9} />
          <StatRow label="F1 Weighted" value={sfr.f1Weighted.toFixed(4)}     good={sfr.f1Weighted > 0.9} />
        </div>
      </div>
    </div>
  );
}
