import styles from './ForecastTimeline.module.css';
import type { ForecastPoint, ForecastSummary } from '@/lib/api';

interface Props {
  forecast: ForecastPoint[];
  summary: ForecastSummary;
}

const SEV_ICON: Record<string, string> = { safe: '🟢', medium: '🟡', severe: '🔴' };

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}

export default function ForecastTimeline({ forecast, summary }: Props) {
  return (
    <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <span className="label">Multi-Hour Forecast</span>
        <h2 className={styles.title}>Upcoming Tide Conditions</h2>
      </div>

      {/* Summary stats */}
      <div className={styles.summaryRow}>
        {summary.bestTime && (
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Best Window</span>
            <span className={`${styles.summaryTime} safe`}>{formatTime(summary.bestTime.timestamp)}</span>
            <span className={styles.summaryLevel}>{summary.bestTime.waterLevel.toFixed(3)} m</span>
          </div>
        )}
        {summary.peakTime && (
          <div className={styles.summaryItem}>
            <span className={styles.summaryLabel}>Peak Level</span>
            <span className={`${styles.summaryTime} severe`}>{formatTime(summary.peakTime.timestamp)}</span>
            <span className={styles.summaryLevel}>{summary.peakTime.waterLevel.toFixed(3)} m</span>
          </div>
        )}
        <div className={styles.summaryItem}>
          <span className={styles.summaryLabel}>Risk Windows</span>
          <span className={styles.summaryCount}>{summary.riskWindows}</span>
          <span className={styles.summaryLevel}>hrs elevated</span>
        </div>
      </div>

      <div className={styles.divider} />

      {/* Hourly forecast cards */}
      <div className={styles.timeline}>
        {forecast.map((point, i) => {
          const sev = point.severity.toLowerCase() as 'safe' | 'medium' | 'severe';
          return (
            <div key={i} className={`${styles.timeCard} ${styles[`card_${sev}`]}`}>
              <span className={styles.timeLabel}>{formatTime(point.timestamp)}</span>
              <span className={styles.timeIcon}>{SEV_ICON[sev] ?? '●'}</span>
              <span className={`${styles.timeLevel} mono`}>{point.waterLevel.toFixed(3)}</span>
              <span className={styles.timeUnit}>m</span>
              <span className={`badge ${sev}`}>{point.severity}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
