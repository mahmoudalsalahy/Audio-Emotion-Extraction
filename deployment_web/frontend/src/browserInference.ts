export type BrowserModel = {
  modelName: string;
  suite: string;
  featureSet: string;
  classNames: string[];
  featureNames: string[];
  standardScaler: {
    mean: number[];
    scale: number[];
  };
  randomForest: {
    nClasses: number;
    nTrees: number;
    trees: TreeModel[];
  };
  knn: {
    nClasses: number;
    nNeighbors: number;
    x: number[][];
    y: number[];
  };
  testMetrics: Record<string, number>;
};

type TreeModel = {
  childrenLeft: number[];
  childrenRight: number[];
  feature: number[];
  threshold: number[];
  value: number[][];
};

export type BrowserPrediction = {
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

const SAMPLE_RATE = 16000;
const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_MFCC = 40;
const N_MELS = 128;
const MIN_DURATION = 0.5;

let modelPromise: Promise<BrowserModel> | null = null;

export function loadBrowserModel() {
  modelPromise ??= fetch(`${import.meta.env.BASE_URL}model/browser-model.json`).then((response) => {
    if (!response.ok) throw new Error('Browser model file was not found.');
    return response.json() as Promise<BrowserModel>;
  });
  return modelPromise;
}

export async function predictInBrowser(file: File, model: BrowserModel): Promise<BrowserPrediction> {
  const samples = await decodeAudio(file);
  const features = extractFeatures(samples);
  const scaled = scaleAndNormalize(features, model);
  const forestScores = predictRandomForest(model.randomForest, scaled);
  const knnScores = predictKnn(model.knn, scaled);
  const scores = blendScores(forestScores, knnScores);
  const order = scores.map((score, index) => ({ score, index })).sort((a, b) => b.score - a.score);
  const top = order[0];
  const second = order[1];
  const confidence = top.score;
  const margin = second ? top.score - second.score : confidence;

  return {
    prediction: model.classNames[top.index] ?? String(top.index),
    confidence,
    margin,
    recommendation: confidence >= 0.6 && margin >= 0.15 ? 'Accept' : 'Review',
    alternatives: order.slice(0, 5).map(({ score, index }) => ({
      label: model.classNames[index] ?? String(index),
      score,
    })),
    model: {
      name: model.modelName,
      suite: model.suite,
      featureSet: model.featureSet,
      version: 'browser-rf',
      selectedFeatures: model.featureNames.length,
      testMetrics: model.testMetrics,
    },
    explanation: {
      summary:
        'Prediction runs fully in the browser using exported RandomForest and KNN components from the saved training model plus JavaScript audio feature extraction.',
      topFeatureNames: model.featureNames.slice(0, 8),
      xaiDataAvailable: false,
    },
  };
}

async function decodeAudio(file: File) {
  const context = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    const samples = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const data = buffer.getChannelData(channel);
      for (let i = 0; i < data.length; i += 1) samples[i] += data[i] / buffer.numberOfChannels;
    }
    return trimSilence(samples);
  } finally {
    await context.close();
  }
}

function trimSilence(samples: Float32Array) {
  let peak = 0;
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
  if (peak === 0) throw new Error('Could not extract valid audio features. Check that the WAV file is readable.');

  const threshold = peak * 0.1;
  let start = 0;
  let end = samples.length - 1;
  while (start < samples.length && Math.abs(samples[start]) < threshold) start += 1;
  while (end > start && Math.abs(samples[end]) < threshold) end -= 1;
  const trimmed = samples.slice(start, end + 1);
  if (trimmed.length / SAMPLE_RATE < MIN_DURATION) throw new Error('Audio is too short after trimming silence.');
  return trimmed;
}

function extractFeatures(samples: Float32Array) {
  const frames = makeFrames(samples);
  const melFilters = createMelFilters();
  const dctMatrix = createDctMatrix();
  const freqBins = Array.from({ length: N_FFT / 2 + 1 }, (_, i) => (i * SAMPLE_RATE) / N_FFT);

  const mfccFrames: number[][] = [];
  const chromaFrames: number[][] = [];
  const spectralCentroid: number[] = [];
  const spectralBandwidth: number[] = [];
  const spectralRolloff: number[] = [];
  const zcr: number[] = [];
  const rms: number[] = [];
  const f0: number[] = [];
  const melDbValues: number[] = [];
  const melEnergyFrames: number[][] = [];

  for (const frame of frames) {
    const spectrum = magnitudeSpectrum(frame);
    const power = spectrum.map((value) => value * value);
    const totalPower = sum(power) || 1e-12;

    const melEnergies = melFilters.map((filter) => {
      let energy = 0;
      for (let i = 0; i < filter.length; i += 1) energy += filter[i] * power[i];
      return Math.max(energy, 1e-12);
    });
    melEnergyFrames.push(melEnergies);

    const chroma = Array.from({ length: 12 }, () => 0);
    for (let i = 1; i < power.length; i += 1) {
      const freq = freqBins[i];
      const midi = Math.round(69 + 12 * Math.log2(freq / 440));
      chroma[((midi % 12) + 12) % 12] += power[i];
    }
    const chromaTotal = sum(chroma) || 1e-12;
    chromaFrames.push(chroma.map((value) => value / chromaTotal));

    const centroid = freqBins.reduce((acc, freq, index) => acc + freq * power[index], 0) / totalPower;
    spectralCentroid.push(centroid);
    spectralBandwidth.push(Math.sqrt(freqBins.reduce((acc, freq, index) => acc + (freq - centroid) ** 2 * power[index], 0) / totalPower));
    spectralRolloff.push(rolloff(freqBins, power, totalPower));
    zcr.push(zeroCrossingRate(frame));
    rms.push(Math.sqrt(sum(Array.from(frame, (value) => value * value)) / frame.length));
    f0.push(estimatePitch(frame));
  }

  const maxMelEnergy = Math.max(...melEnergyFrames.flat(), 1e-12);
  for (const melEnergies of melEnergyFrames) {
    const logMelForMfcc = melEnergies.map((value) => powerToDb(value));
    mfccFrames.push(dctMatrix.map((row) => dot(row, logMelForMfcc)));
    melDbValues.push(...melEnergies.map((value) => Math.max(powerToDb(value / maxMelEnergy), -80)));
  }

  const mfccDelta = delta(mfccFrames);
  const features = [
    ...columnStats(mfccFrames, 'mean'),
    ...columnStats(mfccFrames, 'std'),
    ...columnStats(mfccDelta, 'mean'),
    ...columnStats(mfccDelta, 'std'),
    ...columnStats(chromaFrames, 'mean'),
    ...columnStats(chromaFrames, 'std'),
    mean(spectralCentroid),
    std(spectralCentroid),
    mean(spectralBandwidth),
    std(spectralBandwidth),
    mean(spectralRolloff),
    std(spectralRolloff),
    mean(zcr),
    std(zcr),
    mean(rms),
    std(rms),
    mean(f0),
    std(f0),
    mean(melDbValues),
    std(melDbValues),
    percentile(melDbValues, 25),
    percentile(melDbValues, 75),
  ];

  if (features.length !== 200) throw new Error(`Feature extraction returned ${features.length} features instead of 200.`);
  return features.map((value) => (Number.isFinite(value) ? value : 0));
}

function makeFrames(samples: Float32Array) {
  const frames: Float32Array[] = [];
  const window = hannWindow();
  for (let start = 0; start + N_FFT <= samples.length; start += HOP_LENGTH) {
    const frame = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i += 1) frame[i] = samples[start + i] * window[i];
    frames.push(frame);
  }
  if (!frames.length) {
    const frame = new Float32Array(N_FFT);
    frame.set(samples.slice(0, N_FFT));
    for (let i = 0; i < N_FFT; i += 1) frame[i] *= window[i];
    frames.push(frame);
  }
  return frames;
}

function hannWindow() {
  return Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N_FFT - 1)));
}

function magnitudeSpectrum(frame: Float32Array) {
  const real = Array.from(frame);
  const imag = Array.from({ length: N_FFT }, () => 0);
  fft(real, imag);
  return real.slice(0, N_FFT / 2 + 1).map((value, index) => Math.hypot(value, imag[index]));
}

function fft(real: number[], imag: number[]) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wlenReal = Math.cos(angle);
    const wlenImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wReal = 1;
      let wImag = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uReal = real[i + j];
        const uImag = imag[i + j];
        const vReal = real[i + j + len / 2] * wReal - imag[i + j + len / 2] * wImag;
        const vImag = real[i + j + len / 2] * wImag + imag[i + j + len / 2] * wReal;
        real[i + j] = uReal + vReal;
        imag[i + j] = uImag + vImag;
        real[i + j + len / 2] = uReal - vReal;
        imag[i + j + len / 2] = uImag - vImag;
        [wReal, wImag] = [wReal * wlenReal - wImag * wlenImag, wReal * wlenImag + wImag * wlenReal];
      }
    }
  }
}

function createMelFilters() {
  const hzToMel = (hz: number) => {
    const fMin = 0;
    const fSp = 200 / 3;
    const minLogHz = 1000;
    const minLogMel = (minLogHz - fMin) / fSp;
    const logStep = Math.log(6.4) / 27;
    return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logStep : (hz - fMin) / fSp;
  };
  const melToHz = (mel: number) => {
    const fMin = 0;
    const fSp = 200 / 3;
    const minLogHz = 1000;
    const minLogMel = (minLogHz - fMin) / fSp;
    const logStep = Math.log(6.4) / 27;
    return mel >= minLogMel ? minLogHz * Math.exp(logStep * (mel - minLogMel)) : fMin + fSp * mel;
  };
  const minMel = hzToMel(0);
  const maxMel = hzToMel(SAMPLE_RATE / 2);
  const melPoints = Array.from({ length: N_MELS + 2 }, (_, i) => minMel + (i / (N_MELS + 1)) * (maxMel - minMel));
  const hzPoints = melPoints.map(melToHz);
  const bins = hzPoints.map((hz) => Math.floor(((N_FFT + 1) * hz) / SAMPLE_RATE));

  return Array.from({ length: N_MELS }, (_, m) => {
    const filter = Array.from({ length: N_FFT / 2 + 1 }, () => 0);
    for (let k = bins[m]; k < bins[m + 1]; k += 1) filter[k] = (k - bins[m]) / Math.max(bins[m + 1] - bins[m], 1);
    for (let k = bins[m + 1]; k < bins[m + 2]; k += 1) filter[k] = (bins[m + 2] - k) / Math.max(bins[m + 2] - bins[m + 1], 1);
    const enorm = 2 / Math.max(hzPoints[m + 2] - hzPoints[m], 1e-12);
    for (let k = 0; k < filter.length; k += 1) filter[k] *= enorm;
    return filter;
  });
}

function createDctMatrix() {
  return Array.from({ length: N_MFCC }, (_, i) => {
    const scale = i === 0 ? Math.sqrt(1 / (4 * N_MELS)) * 2 : Math.sqrt(1 / (2 * N_MELS)) * 2;
    return Array.from({ length: N_MELS }, (_, j) => scale * Math.cos((Math.PI * i * (j + 0.5)) / N_MELS));
  });
}

function delta(frames: number[][]) {
  return frames.map((_, frameIndex) =>
    frames[0].map((__, column) => {
      let numerator = 0;
      let denominator = 0;
      for (let n = 1; n <= 2; n += 1) {
        const prev = frames[Math.max(0, frameIndex - n)][column];
        const next = frames[Math.min(frames.length - 1, frameIndex + n)][column];
        numerator += n * (next - prev);
        denominator += 2 * n * n;
      }
      return numerator / denominator;
    }),
  );
}

function estimatePitch(frame: Float32Array) {
  const minLag = Math.floor(SAMPLE_RATE / 500);
  const maxLag = Math.floor(SAMPLE_RATE / 50);
  let bestLag = 0;
  let bestScore = 0;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let score = 0;
    for (let i = 0; i < frame.length - lag; i += 1) score += frame[i] * frame[i + lag];
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }
  return bestLag ? SAMPLE_RATE / bestLag : 0;
}

function predictRandomForest(forest: BrowserModel['randomForest'], features: number[]) {
  const scores = Array.from({ length: forest.nClasses }, () => 0);
  for (const tree of forest.trees) {
    let node = 0;
    while (tree.childrenLeft[node] !== -1) {
      node = features[tree.feature[node]] <= tree.threshold[node] ? tree.childrenLeft[node] : tree.childrenRight[node];
    }
    const values = tree.value[node];
    const total = sum(values) || 1e-12;
    for (let i = 0; i < scores.length; i += 1) scores[i] += values[i] / total;
  }
  const total = sum(scores) || 1e-12;
  return scores.map((score) => score / total);
}

function predictKnn(knn: BrowserModel['knn'], features: number[]) {
  const nearest: Array<{ distance: number; label: number }> = [];
  for (let rowIndex = 0; rowIndex < knn.x.length; rowIndex += 1) {
    const candidate = knn.x[rowIndex];
    let distance = 0;
    for (let featureIndex = 0; featureIndex < candidate.length; featureIndex += 1) {
      const diff = features[featureIndex] - candidate[featureIndex];
      distance += diff * diff;
    }
    nearest.push({ distance: Math.sqrt(distance), label: knn.y[rowIndex] });
  }
  nearest.sort((a, b) => a.distance - b.distance);

  const scores = Array.from({ length: knn.nClasses }, () => 0);
  for (const neighbor of nearest.slice(0, knn.nNeighbors)) {
    scores[neighbor.label] += 1 / Math.max(neighbor.distance, 1e-9);
  }
  const total = sum(scores) || 1e-12;
  return scores.map((score) => score / total);
}

function blendScores(forestScores: number[], knnScores: number[]) {
  const scores = forestScores.map((score, index) => score * 0.65 + knnScores[index] * 0.35);
  const total = sum(scores) || 1e-12;
  return scores.map((score) => score / total);
}

function scaleAndNormalize(features: number[], model: BrowserModel) {
  const scaled = features.map((value, index) => (value - model.standardScaler.mean[index]) / model.standardScaler.scale[index]);
  const norm = Math.sqrt(sum(scaled.map((value) => value * value))) || 1;
  return scaled.map((value) => value / norm);
}

function columnStats(rows: number[][], mode: 'mean' | 'std') {
  return rows[0].map((_, index) => {
    const values = rows.map((row) => row[index]);
    return mode === 'mean' ? mean(values) : std(values);
  });
}

function rolloff(freqs: number[], power: number[], totalPower: number) {
  const target = totalPower * 0.85;
  let running = 0;
  for (let i = 0; i < power.length; i += 1) {
    running += power[i];
    if (running >= target) return freqs[i];
  }
  return freqs[freqs.length - 1];
}

function zeroCrossingRate(frame: Float32Array) {
  let crossings = 0;
  for (let i = 1; i < frame.length; i += 1) {
    if ((frame[i - 1] >= 0 && frame[i] < 0) || (frame[i - 1] < 0 && frame[i] >= 0)) crossings += 1;
  }
  return crossings / Math.max(frame.length - 1, 1);
}

function percentile(values: number[], p: number) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = ((sorted.length - 1) * p) / 100;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function powerToDb(value: number) {
  return 10 * Math.log10(Math.max(value, 1e-10));
}

function mean(values: number[]) {
  return sum(values) / Math.max(values.length, 1);
}

function std(values: number[]) {
  const avg = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)));
}

function sum(values: number[]) {
  return values.reduce((acc, value) => acc + value, 0);
}

function dot(left: number[], right: number[]) {
  let total = 0;
  for (let i = 0; i < left.length; i += 1) total += left[i] * right[i];
  return total;
}
