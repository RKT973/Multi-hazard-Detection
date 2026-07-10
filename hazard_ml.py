from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Mapping, Optional

import joblib
import numpy as np


SENSOR_KEYS = ("temperature", "humidity", "gas", "light", "pressure")
MODEL_PATH = Path(__file__).resolve().parent / "random_forest_hazard_model.joblib"

ADC_STD_FLOOR = 10.0
FLOAT_STD_FLOOR = 0.5
PRESSURE_STD_FLOOR = 0.5

CLASS_LABELS = ("NO_RISK", "FIRE_RISK", "SMOKE_RISK", "WATER_LEAK_RISK")


def _std_floor(sensor: str) -> float:
    if sensor in {"gas", "light"}:
        return ADC_STD_FLOOR
    if sensor == "pressure":
        return PRESSURE_STD_FLOOR
    return FLOAT_STD_FLOOR


def _stats(values: Iterable[float], sensor: str) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, _std_floor(sensor)
    mean = float(arr.mean())
    std = float(arr.std())
    return mean, max(std, _std_floor(sensor))


def _z(value: float, mean: float, std: float) -> float:
    return (float(value) - mean) / std


def _positive_anomaly(z_score: float, k: float) -> float:
    if z_score < k:
        return 0.0
    return min(1.0, (z_score - k) / k)


def build_deviation_features(
    reading: Mapping[str, float],
    history: Mapping[str, Iterable[float]],
) -> Dict[str, float]:
    """Build model inputs from deviations, not raw sensor values."""

    features: Dict[str, float] = {}
    thresholds = {
        "temperature": 2.0,
        "humidity": 2.0,
        "gas": 2.5,
        "light": 2.0,
        "pressure": 2.0,
    }

    for sensor in SENSOR_KEYS:
        mean, std = _stats(history.get(sensor, ()), sensor)
        value = float(reading.get(sensor, 1013.25 if sensor == "pressure" else 0.0))
        z_score = _z(value, mean, std)
        k = thresholds[sensor]

        features[f"{sensor}_z"] = z_score
        features[f"{sensor}_rise"] = _positive_anomaly(z_score, k)
        features[f"{sensor}_drop"] = _positive_anomaly(-z_score, k)
        features[f"{sensor}_abs"] = min(1.0, abs(z_score) / (k * 2.0))

    features["fire_pattern"] = (
        0.35 * features["temperature_rise"]
        + 0.30 * features["gas_rise"]
        + 0.20 * features["humidity_drop"]
        + 0.15 * features["pressure_abs"]
    )
    features["smoke_pattern"] = (
        0.40 * features["gas_rise"]
        + 0.35 * features["light_drop"]
        + 0.25 * features["humidity_rise"]
    )
    features["water_pattern"] = (
        0.75 * features["humidity_rise"]
        + 0.25 * features["pressure_rise"]
    )
    return features


FEATURE_COLUMNS = tuple(
    [f"{sensor}_{kind}" for sensor in SENSOR_KEYS for kind in ("z", "rise", "drop", "abs")]
    + ["fire_pattern", "smoke_pattern", "water_pattern"]
)


def features_to_array(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray([[float(features[name]) for name in FEATURE_COLUMNS]], dtype=float)


class HazardRandomForest:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        if model_path.exists():
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.feature_columns = tuple(bundle["feature_columns"])
            self.class_labels = tuple(bundle.get("model_classes", self.model.classes_))
        else:
            self.feature_columns = FEATURE_COLUMNS
            self.class_labels = CLASS_LABELS

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(
        self,
        reading: Mapping[str, float],
        history: Mapping[str, Iterable[float]],
    ) -> Dict[str, object]:
        if not self.ready:
            return {"hazard": "SAFE", "risk": "NO_RISK", "confidence": 0.0, "features": {}}

        features = build_deviation_features(reading, history)
        row = np.asarray([[float(features[name]) for name in self.feature_columns]], dtype=float)
        probabilities = self.model.predict_proba(row)[0]
        best_index = int(np.argmax(probabilities))
        risk = str(self.class_labels[best_index])

        return {
            "hazard": self._risk_to_hazard(risk),
            "risk": risk,
            "confidence": round(float(probabilities[best_index]), 4),
            "probabilities": {
                str(self.class_labels[i]): round(float(probabilities[i]), 4)
                for i in range(len(self.class_labels))
            },
            "features": {key: round(value, 4) for key, value in features.items()},
        }

    @staticmethod
    def _risk_to_hazard(risk: str) -> str:
        return {
            "NO_RISK": "SAFE",
            "FIRE_RISK": "FIRE_RISK",
            "SMOKE_RISK": "SMOKE_RISK",
            "WATER_LEAK_RISK": "WATER_LEAK_RISK",
        }.get(risk, "SAFE")


class RollingBaseline:
    def __init__(self, window_size: int = 20):
        self.history: Dict[str, Deque[float]] = {
            sensor: deque(maxlen=window_size) for sensor in SENSOR_KEYS
        }

    def update(self, reading: Mapping[str, float]) -> None:
        for sensor in SENSOR_KEYS:
            self.history[sensor].append(float(reading.get(sensor, 0.0)))

    def as_history(self) -> Dict[str, Iterable[float]]:
        return self.history

    def is_ready(self) -> bool:
        return all(len(values) == values.maxlen for values in self.history.values())


def predict_from_series(
    model: HazardRandomForest,
    reading: Mapping[str, float],
    baseline: RollingBaseline,
) -> Optional[Dict[str, object]]:
    if not baseline.is_ready():
        baseline.update(reading)
        return None

    prediction = model.predict(reading, baseline.as_history())
    baseline.update(reading)
    return prediction
