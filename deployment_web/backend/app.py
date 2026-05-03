from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_ROOT / "models" / "model-v1.pkl"
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else PROJECT_ROOT / "models" / "model-v1.pkl",
    )
)
XAI_DATA_PATH = Path(os.getenv("XAI_DATA_PATH", PROJECT_ROOT / "data" / "xai-data.pkl"))

SAMPLE_RATE = 16000
MIN_DURATION = 0.5
TRIM_DB = 20
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
FEATURE_DIM = 202


def cors_origins() -> list[str]:
    configured_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *configured_origins,
    ]


app = FastAPI(
    title="Speech Emotion Recognition Deployment API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_audio(filepath: str) -> tuple[np.ndarray | None, int, float]:
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
        duration = len(y) / sr
        if duration < MIN_DURATION:
            return None, sr, duration
        return y, sr, duration
    except Exception:
        return None, SAMPLE_RATE, 0.0


def extract_features(filepath: str) -> np.ndarray:
    try:
        y, sr = preprocess_audio(filepath)[:2]
        if y is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        if len(f0_clean) == 0:
            f0_clean = np.array([0.0])

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        features = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(mfcc_delta, axis=1),
                np.std(mfcc_delta, axis=1),
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                [np.mean(sc), np.std(sc)],
                [np.mean(sb), np.std(sb)],
                [np.mean(sro), np.std(sro)],
                [np.mean(zcr), np.std(zcr)],
                [np.mean(rms), np.std(rms)],
                [np.mean(f0_clean), np.std(f0_clean)],
                [
                    np.mean(mel_db),
                    np.std(mel_db),
                    np.percentile(mel_db, 25),
                    np.percentile(mel_db, 75),
                ],
            ]
        )
        return features.astype(np.float32)
    except Exception:
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def load_model_package() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/model-v1.pkl was not found. Run the notebook training and save-model cell first."
        )

    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def probability_like_scores(model: Any, x_matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_matrix), dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_matrix), dtype=float)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)

    prediction = model.predict(x_matrix)
    n_classes = len(getattr(model, "classes_", []))
    if n_classes == 0:
        n_classes = 1
    scores = np.zeros((len(prediction), n_classes), dtype=float)
    for row_idx, label in enumerate(prediction):
        label_idx = int(label)
        if 0 <= label_idx < n_classes:
            scores[row_idx, label_idx] = 1.0
    return scores


def predict_file(filepath: str) -> dict[str, Any]:
    package = load_model_package()
    model = package["model"]
    scaler = package["scaler"]
    feature_mask = np.asarray(package["feature_mask"], dtype=bool)
    class_names = package.get("class_names") or [str(c) for c in getattr(model, "classes_", [])]

    raw_features = extract_features(filepath).reshape(1, -1)
    expected_dim = len(package.get("all_feature_names", [])) or FEATURE_DIM
    if raw_features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected_dim}, extractor returned {raw_features.shape[1]}."
        )
    if np.all(raw_features == 0):
        raise ValueError("Could not extract valid audio features. Check that the file is a readable WAV clip.")

    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = scaler.transform(raw_features)
    selected = scaled[:, feature_mask]

    prediction_id = int(model.predict(selected)[0])
    scores = probability_like_scores(model, selected)[0]
    order = np.argsort(scores)[::-1]

    prediction = class_names[prediction_id] if prediction_id < len(class_names) else str(prediction_id)
    confidence = float(scores[prediction_id]) if prediction_id < len(scores) else float(np.max(scores))
    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else confidence
    recommendation = "Accept" if confidence >= 0.60 and margin >= 0.15 else "Review"

    alternatives = [
        {
            "label": class_names[idx] if idx < len(class_names) else str(idx),
            "score": float(scores[idx]),
        }
        for idx in order[: min(5, len(order))]
    ]

    top_features = package.get("feature_names", [])[: min(8, len(package.get("feature_names", [])))]
    return {
        "prediction": prediction,
        "confidence": confidence,
        "margin": margin,
        "recommendation": recommendation,
        "alternatives": alternatives,
        "model": {
            "name": package.get("model_name", "Unknown"),
            "suite": package.get("suite", "Unknown"),
            "featureSet": package.get("feature_set", "Unknown"),
            "version": package.get("version", "Unknown"),
            "selectedFeatures": int(feature_mask.sum()),
            "testMetrics": package.get("test_metrics", {}),
        },
        "explanation": {
            "summary": (
                "Prediction is produced by the saved training artifact after the same "
                "audio feature extraction, scaling, and feature selection steps used in the notebook."
            ),
            "topFeatureNames": top_features,
            "xaiDataAvailable": XAI_DATA_PATH.exists(),
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model_available = MODEL_PATH.exists()
    payload: dict[str, Any] = {
        "status": "ready" if model_available else "model_missing",
        "modelPath": str(MODEL_PATH),
        "modelAvailable": model_available,
        "xaiDataAvailable": XAI_DATA_PATH.exists(),
    }

    if model_available:
        try:
            package = load_model_package()
            payload["model"] = {
                "name": package.get("model_name", "Unknown"),
                "suite": package.get("suite", "Unknown"),
                "featureSet": package.get("feature_set", "Unknown"),
                "version": package.get("version", "Unknown"),
            }
        except Exception as exc:
            payload["status"] = "model_error"
            payload["error"] = str(exc)

    return payload


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload a .wav audio file.")

    suffix = Path(file.filename).suffix or ".wav"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())

        result = predict_file(temp_path)
        result["filename"] = file.filename
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
