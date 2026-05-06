import { fetchDashboard } from '@/lib/api';
import HeroHeader from '@/components/HeroHeader';
import SeverityCard from '@/components/SeverityCard';
import MetricsGrid from '@/components/MetricsGrid';
import TideChart from '@/components/TideChart';
import ForecastTimeline from '@/components/ForecastTimeline';
import ConfidencePanel from '@/components/ConfidencePanel';
import ModelStatsPanel from '@/components/ModelStatsPanel';
import InfoPanel from '@/components/InfoPanel';
import styles from './page.module.css';

export const revalidate = 300; // Re-fetch every 5 minutes (ISR)

export default async function HomePage() {
  let data;
  let error: string | null = null;

  try {
    data = await fetchDashboard(72, 6);
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to connect to backend';
  }

  if (error || !data) {
    return (
      <div className="page">
        <div className={styles.errorWrap}>
          <div className={styles.errorIcon}>🌊</div>
          <h1 className={styles.errorTitle}>Coastal Sentinel</h1>
          <div className="error-banner">
            <span>⚠</span>
            <span>Unable to reach the backend API. {error}</span>
          </div>
          <p className={styles.errorHint}>
            Make sure the FastAPI backend is running at{' '}
            <code>{process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      {/* Header + alert banner */}
      <HeroHeader location={data.location} alerts={data.alerts} />

      {/* Row 1: Severity card (wide) + Metrics */}
      <div className={styles.row1}>
        <SeverityCard prediction={data.prediction} />
        <div className={styles.metricsWrap}>
          <MetricsGrid metrics={data.metrics} />
        </div>
      </div>

      {/* Row 2: Tide chart (full width) */}
      <div className={styles.fullRow}>
        <TideChart trend={data.trend} />
      </div>

      {/* Row 3: Forecast timeline (full width) */}
      <div className={styles.fullRow}>
        <ForecastTimeline forecast={data.futureForecast} summary={data.forecastSummary} />
      </div>

      {/* Row 4: Confidence + Model stats + Info */}
      <div className={styles.row4}>
        <ConfidencePanel confidence={data.confidence} />
        <ModelStatsPanel performance={data.modelPerformance} />
        <InfoPanel info={data.info} location={data.location} />
      </div>

      {/* Footer */}
      <footer className={styles.footer}>
        <span>Coastal Sentinel &copy; {new Date().getFullYear()}</span>
        <span className={styles.footerDot} />
        <span>Powered by PyTorch LSTM · Open-Meteo API</span>
        <span className={styles.footerDot} />
        <span className={styles.footerPulse} /> Live
      </footer>
    </div>
  );
}
