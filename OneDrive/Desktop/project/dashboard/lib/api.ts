const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export interface Location {
  name: string;
  region: string;
  stationId: string;
  latitude: number;
  longitude: number;
  latestObservationTime: string;
  modelWindowEnd: string;
  lookbackHours: number;
}

export interface Prediction {
  severity: string;
  severityColor: string;
  predictedTideLevel: number;
  unit: string;
  lastUpdated: string;
  summary: string;
  classifierSeverity: string;
  observedWaterLevel: number;
}

export interface Confidence {
  score: number;
  label: string;
  classification: string;
  note: string;
  probabilities: {
    safe: number;
    medium: number;
    severe: number;
  };
}

export interface Metric {
  label: string;
  value: number;
  unit: string;
  icon: string;
}

export interface Trend {
  observed: number[];
  predicted: number[];
  observedTimestamps: string[];
  predictedTimestamps: string[];
  labels: string[];
}

export interface ForecastPoint {
  timestamp: string;
  waterLevel: number;
  severity: string;
}

export interface ForecastSummary {
  bestTime: ForecastPoint | null;
  dangerousTime: ForecastPoint | null;
  peakTime: ForecastPoint | null;
  riskWindows: number;
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  f1Macro: number;
  f1Weighted: number;
}

export interface ModelPerformance {
  regression: ModelMetrics;
  classification: ClassificationMetrics;
  severityFromRegression: ClassificationMetrics;
}

export interface Alerts {
  severity: string;
  headline: string;
  message: string;
}

export interface Info {
  title: string;
  description: string;
  points: string[];
}

export interface DashboardData {
  location: Location;
  prediction: Prediction;
  confidence: Confidence;
  metrics: Metric[];
  trend: Trend;
  futureForecast: ForecastPoint[];
  forecastSummary: ForecastSummary;
  modelPerformance: ModelPerformance;
  alerts: Alerts;
  info: Info;
}

export async function fetchDashboard(
  lookbackHours = 72,
  forecastHours = 6,
): Promise<DashboardData> {
  const url = `${API_URL}/api/dashboard/live?lookback_hours=${lookbackHours}&forecast_hours=${forecastHours}`;
  const res = await fetch(url, { next: { revalidate: 300 } });
  if (!res.ok) {
    throw new Error(`Backend returned ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<DashboardData>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/api/health`, { cache: 'no-store' });
    return res.ok;
  } catch {
    return false;
  }
}
