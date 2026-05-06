import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Coastal Sentinel — Live Tide & Flood Risk Dashboard',
  description:
    'Real-time AI-powered coastal monitoring for Digha Beach, West Bengal. Live tide level predictions, severity classification, and multi-hour forecasts powered by PyTorch LSTM models.',
  keywords: ['tide prediction', 'coastal monitoring', 'flood risk', 'Digha Beach', 'LSTM', 'AI'],
  openGraph: {
    title: 'Coastal Sentinel',
    description: 'Live AI-powered tide predictions for Digha Beach',
    type: 'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="theme-color" content="#050d1a" />
      </head>
      <body>{children}</body>
    </html>
  );
}
