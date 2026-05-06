import styles from './InfoPanel.module.css';
import type { Info, Location } from '@/lib/api';

interface Props { info: Info; location: Location; }

export default function InfoPanel({ info, location }: Props) {
  return (
    <div className={`card ${styles.card}`}>
      <span className="label">About</span>
      <h2 className={styles.title}>{info.title}</h2>
      <p className={styles.description}>{info.description}</p>
      <ul className={styles.points}>
        {info.points.map((pt, i) => (
          <li key={i} className={styles.point}>
            <span className={styles.pointIcon}>◆</span>
            {pt}
          </li>
        ))}
      </ul>
      <div className={styles.divider} />
      <div className={styles.meta}>
        <div className={styles.metaItem}>
          <span className="label">Location</span>
          <span className={styles.metaValue}>{location.name}</span>
        </div>
        <div className={styles.metaItem}>
          <span className="label">Lookback Window</span>
          <span className={styles.metaValue}>{location.lookbackHours}h</span>
        </div>
        <div className={styles.metaItem}>
          <span className="label">Model Window End</span>
          <span className={styles.metaValue}>{location.modelWindowEnd}</span>
        </div>
      </div>
    </div>
  );
}
