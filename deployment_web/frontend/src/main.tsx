import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  AlertCircle,
  AudioWaveform,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  FileAudio,
  Gauge,
  Loader2,
  ShieldCheck,
  UploadCloud,
} from 'lucide-react';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

type HealthResponse = {
  status: string;
  modelAvailable: boolean;
  xaiDataAvailable: boolean;
  model?: {
    name: string;
    suite: string;
    featureSet: string;
    version: string;
  };
  error?: string;
};

type PredictionResponse = {
  filename: string;
  prediction: string;
  confidence: number;
  margin: number;
  recommendation: string;
  alternatives: Array<{ label: string; score: number }>;
  model: {
    name: string;
    suite: string;
    featureSet: string;
    version: string;
    selectedFeatures: number;
    testMetrics: Record<string, number>;
  };
  explanation: {
    summary: string;
    topFeatureNames: string[];
    xaiDataAvailable: boolean;
  };
};

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function IconPanel({ children }: { children: React.ReactNode }) {
  return <div className="icon-panel">{children}</div>;
}

function MetricCard({
  icon,
  label,
  value,
  accent = 'blue',
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  accent?: 'blue' | 'green' | 'amber';
}) {
  return (
    <section className={`metric-card ${accent}`}>
      <div className="metric-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
      </div>
    </section>
  );
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);

  const fetchHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const data = (await response.json()) as HealthResponse;
      setHealth(data);
    } catch {
      setHealth({
        status: 'api_offline',
        modelAvailable: false,
        xaiDataAvailable: false,
        error: 'API is not reachable',
      });
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  const status = useMemo(() => {
    if (!health) return { label: 'Checking API', className: 'neutral' };
    if (health.status === 'ready') return { label: 'Model ready', className: 'ready' };
    if (health.status === 'api_offline') return { label: 'API offline', className: 'danger' };
    return { label: 'Model missing', className: 'warning' };
  }, [health]);

  const chooseFile = (candidate?: File | null) => {
    setResult(null);
    setError('');
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith('.wav')) {
      setError('Please choose a WAV audio file.');
      setFile(null);
      return;
    }
    setFile(candidate);
  };

  const submit = async () => {
    if (!file) {
      setError('Choose a WAV audio file before running prediction.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }
      setResult(data as PredictionResponse);
      fetchHealth();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="side-rail">
          <IconPanel>
            <AudioWaveform size={28} />
          </IconPanel>
          <div className="rail-line" />
          <IconPanel>
            <BrainCircuit size={24} />
          </IconPanel>
          <IconPanel>
            <ShieldCheck size={24} />
          </IconPanel>
        </aside>

        <section className="main-panel">
          <header className="topbar">
            <div>
              <p className="eyebrow">CREMA-D Speech Emotion Recognition</p>
              <h1>Model Deployment Console</h1>
            </div>
            <div className={`status-pill ${status.className}`}>
              {status.className === 'ready' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
              <span>{status.label}</span>
            </div>
          </header>

          <section className="hero-grid">
            <div className="upload-section">
              <div
                className={`dropzone ${dragging ? 'dragging' : ''}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  chooseFile(event.dataTransfer.files.item(0));
                }}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click();
                }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept=".wav,audio/wav"
                  onChange={(event) => chooseFile(event.target.files?.item(0))}
                />
                <div className="upload-orbit">
                  <UploadCloud size={34} />
                </div>
                <div>
                  <h2>{file ? file.name : 'Upload WAV sample'}</h2>
                  <p>{file ? `${(file.size / 1024).toFixed(1)} KB ready for prediction` : 'Drag an audio file here or select one from disk'}</p>
                </div>
              </div>

              <div className="action-row">
                <button className="primary-button" onClick={submit} disabled={loading || !file}>
                  {loading ? <Loader2 className="spin" size={18} /> : <Activity size={18} />}
                  <span>{loading ? 'Analyzing audio' : 'Predict emotion'}</span>
                </button>
                <button className="ghost-button" onClick={fetchHealth}>
                  <Gauge size={18} />
                  <span>Refresh status</span>
                </button>
              </div>

              {error && (
                <div className="error-box">
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}
            </div>

            <div className="model-card">
              <div className="card-header">
                <div>
                  <p className="eyebrow">Saved artifact</p>
                  <h2>{health?.model?.name ?? 'Waiting for model'}</h2>
                </div>
                <BrainCircuit size={26} />
              </div>
              <dl>
                <div>
                  <dt>Suite</dt>
                  <dd>{health?.model?.suite ?? 'Not loaded'}</dd>
                </div>
                <div>
                  <dt>Feature set</dt>
                  <dd>{health?.model?.featureSet ?? 'Run training save cell'}</dd>
                </div>
                <div>
                  <dt>XAI data</dt>
                  <dd>{health?.xaiDataAvailable ? 'Available' : 'Not saved yet'}</dd>
                </div>
              </dl>
            </div>
          </section>

          <section className="results-zone">
            {result ? (
              <>
                <div className="metrics-grid">
                  <MetricCard icon={<BrainCircuit size={20} />} label="Prediction" value={result.prediction} />
                  <MetricCard icon={<Gauge size={20} />} label="Confidence" value={percent(result.confidence)} accent="green" />
                  <MetricCard icon={<BarChart3 size={20} />} label="Margin" value={percent(result.margin)} accent="amber" />
                </div>

                <div className="detail-grid">
                  <section className="analysis-card">
                    <div className="card-header">
                      <div>
                        <p className="eyebrow">Ranking</p>
                        <h2>Class alternatives</h2>
                      </div>
                      <BarChart3 size={24} />
                    </div>
                    <div className="bars">
                      {result.alternatives.map((item) => (
                        <div className="bar-row" key={item.label}>
                          <span>{item.label}</span>
                          <div className="bar-track">
                            <div className="bar-fill" style={{ width: `${Math.max(item.score * 100, 2)}%` }} />
                          </div>
                          <strong>{percent(item.score)}</strong>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="analysis-card">
                    <div className="card-header">
                      <div>
                        <p className="eyebrow">Reasoning</p>
                        <h2>{result.recommendation} prediction</h2>
                      </div>
                      <ShieldCheck size={24} />
                    </div>
                    <p className="summary-text">{result.explanation.summary}</p>
                    <div className="feature-list">
                      {result.explanation.topFeatureNames.length ? (
                        result.explanation.topFeatureNames.map((name) => <span key={name}>{name}</span>)
                      ) : (
                        <span>Feature names will appear after the saved artifact is loaded</span>
                      )}
                    </div>
                  </section>
                </div>
              </>
            ) : (
              <section className="empty-state">
                <FileAudio size={42} />
                <h2>No prediction yet</h2>
                <p>Upload a WAV file and run the saved model generated from notebook training.</p>
              </section>
            )}
          </section>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
