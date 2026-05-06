import styles from './MetricsGrid.module.css';
import type { Metric } from '@/lib/api';

interface Props { metrics: Metric[]; }

const ICON_MAP: Record<string, string> = {
  wind: '💨',
  pressure: '🌡',
  air: '🌬',
  water: '🌊',
};

function MetricCard({ metric }: { metric: Metric }) {
  return (
    <div className={`card ${styles.metricCard}`}>
      <div className={styles.iconWrap}>
        <span className={styles.icon}>{ICON_MAP[metric.icon] ?? '📊'}</span>
      </div>
      <div className={styles.content}>
        <span className="label">{metric.label}</span>
        <div className={styles.valueRow}>
          <span className={`${styles.metricValue} mono`}>{metric.value}</span>
          <span className={styles.metricUnit}>{metric.unit}</span>
        </div>
      </div>
      <div className={styles.accent} />
    </div>
  );
}

export default function MetricsGrid({ metrics }: Props) {
  return (
    <div className={styles.grid}>
      {metrics.map((m) => (
        <MetricCard key={m.label} metric={m} />
      ))}
    </div>
  );
}
