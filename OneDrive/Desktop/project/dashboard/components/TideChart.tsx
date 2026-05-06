'use client';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import styles from './TideChart.module.css';
import type { Trend } from '@/lib/api';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

interface Props { trend: Trend; }

function formatLabel(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}

export default function TideChart({ trend }: Props) {
  const obsLabels  = trend.observedTimestamps.map(formatLabel);
  const predLabels = trend.predictedTimestamps.map(formatLabel);
  const allLabels  = [...obsLabels, ...predLabels.slice(1)];

  const obsLen  = trend.observed.length;
  const predLen = trend.predicted.length;
  const totalLen = obsLen + predLen - 1;

  // Pad observed series with nulls where predicted takes over
  const obsData  = [...trend.observed, ...Array(predLen - 1).fill(null)];
  // Pad predicted series with nulls before the handoff point
  const predData = [...Array(obsLen - 1).fill(null), ...trend.predicted];

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        align: 'end',
        labels: {
          color: '#94a3b8',
          font: { family: 'Inter', size: 12 },
          boxWidth: 24,
          padding: 16,
          usePointStyle: true,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(10, 22, 40, 0.95)',
        borderColor: 'rgba(56, 139, 253, 0.3)',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        padding: 12,
        callbacks: {
          label: (ctx) =>
            ctx.parsed.y !== null
              ? `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)} m`
              : '',
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(56, 139, 253, 0.06)' },
        ticks: {
          color: '#4a6080',
          font: { family: 'JetBrains Mono', size: 11 },
          maxRotation: 0,
          maxTicksLimit: 12,
        },
        border: { color: 'rgba(56, 139, 253, 0.1)' },
      },
      y: {
        grid: { color: 'rgba(56, 139, 253, 0.06)' },
        ticks: {
          color: '#4a6080',
          font: { family: 'JetBrains Mono', size: 11 },
          callback: (v) => `${Number(v).toFixed(2)}m`,
        },
        border: { color: 'rgba(56, 139, 253, 0.1)' },
      },
    },
    elements: {
      point: { radius: 0, hitRadius: 8, hoverRadius: 4 },
      line: { tension: 0.4, borderWidth: 2 },
    },
  };

  const data = {
    labels: allLabels,
    datasets: [
      {
        label: 'Observed',
        data: obsData,
        borderColor: '#22d3ee',
        backgroundColor: 'rgba(34, 211, 238, 0.08)',
        fill: true,
        spanGaps: false,
      },
      {
        label: 'Forecast',
        data: predData,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.08)',
        fill: true,
        borderDash: [6, 3],
        spanGaps: false,
      },
    ],
  };

  return (
    <div className={`card ${styles.card}`}>
      <div className={styles.header}>
        <div>
          <span className="label">Water Level Trend</span>
          <h2 className={styles.title}>Observed & Forecast</h2>
        </div>
        <div className={styles.badges}>
          <span className={styles.badge}>
            <span className={styles.dotCyan} /> Observed
          </span>
          <span className={styles.badge}>
            <span className={styles.dotIndigo} /> Forecast
          </span>
        </div>
      </div>
      <div className={styles.chartWrap}>
        <Line options={options} data={data} />
      </div>
    </div>
  );
}
