import styles from './HeroHeader.module.css';
import type { Location, Alerts } from '@/lib/api';

interface Props {
  location: Location;
  alerts: Alerts;
}

const SEVERITY_ICON: Record<string, string> = {
  safe: '✦',
  medium: '⚠',
  severe: '⚡',
};

export default function HeroHeader({ location, alerts }: Props) {
  const sev = alerts.severity.toLowerCase() as 'safe' | 'medium' | 'severe';
  return (
    <header className={styles.header}>
      {/* Top bar */}
      <div className={styles.topBar}>
        <div className={styles.brand}>
          <span className={styles.brandIcon}>🌊</span>
          <span className={styles.brandName}>Coastal Sentinel</span>
          <span className={styles.brandTag}>LIVE</span>
        </div>
        <div className={styles.meta}>
          <span className={styles.metaItem}>
            <span className={styles.metaLabel}>Station</span>
            <span className={styles.metaValue}>{location.stationId}</span>
          </span>
          <span className={styles.metaDivider} />
          <span className={styles.metaItem}>
            <span className={styles.metaLabel}>Updated</span>
            <span className={styles.metaValue}>{location.latestObservationTime}</span>
          </span>
          <span className={styles.metaDivider} />
          <span className={styles.metaItem}>
            <span className={styles.metaLabel}>Coords</span>
            <span className={styles.metaValue}>
              {location.latitude.toFixed(3)}°N, {location.longitude.toFixed(3)}°E
            </span>
          </span>
        </div>
      </div>

      {/* Hero title */}
      <div className={styles.heroContent}>
        <div className={styles.locationInfo}>
          <h1 className={styles.locationName}>{location.name}</h1>
          <p className={styles.locationRegion}>{location.region}</p>
        </div>
      </div>

      {/* Alert banner */}
      <div className={`${styles.alertBanner} ${styles[sev]}`}>
        <span className={styles.alertIcon}>{SEVERITY_ICON[sev] ?? '●'}</span>
        <div className={styles.alertText}>
          <strong>{alerts.headline}</strong>
          <span>{alerts.message}</span>
        </div>
        <div className={`${styles.pulse} ${styles[`pulse_${sev}`]}`} />
      </div>
    </header>
  );
}
