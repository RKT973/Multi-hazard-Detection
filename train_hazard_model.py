from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from hazard_ml import CLASS_LABELS, FEATURE_COLUMNS, MODEL_PATH, build_deviation_features


RNG = np.random.default_rng(42)


BASELINE_MEANS = {
    "temperature": 28.0,
    "humidity": 58.0,
    "gas": 330.0,
    "light": 650.0,
    "pressure": 1013.25,
}

BASELINE_STD = {
    "temperature": 1.2,
    "humidity": 4.0,
    "gas": 35.0,
    "light": 90.0,
    "pressure": 1.2,
}


def make_baseline() -> Dict[str, List[float]]:
    means = {
        sensor: mean + RNG.normal(0, BASELINE_STD[sensor] * 0.35)
        for sensor, mean in BASELINE_MEANS.items()
    }
    return {
        sensor: RNG.normal(means[sensor], BASELINE_STD[sensor], 20).tolist()
        for sensor in BASELINE_MEANS
    }


def latest_from_baseline(history: Dict[str, List[float]]) -> Dict[str, float]:
    return {sensor: float(values[-1]) for sensor, values in history.items()}


def inject(label: str, reading: Dict[str, float], history: Dict[str, List[float]]) -> Dict[str, float]:
    means = {sensor: float(np.mean(values)) for sensor, values in history.items()}
    stds = {sensor: max(float(np.std(values)), 0.5) for sensor, values in history.items()}

    if label == "NO_RISK":
        return reading

    if label == "FIRE_RISK":
        reading["temperature"] = means["temperature"] + RNG.uniform(3.0, 6.5) * stds["temperature"]
        reading["gas"] = means["gas"] + RNG.uniform(3.0, 6.0) * max(stds["gas"], 10.0)
        reading["humidity"] = means["humidity"] - RNG.uniform(2.3, 5.0) * stds["humidity"]
        reading["pressure"] = means["pressure"] + RNG.choice([-1, 1]) * RNG.uniform(2.0, 4.5) * max(stds["pressure"], 0.5)
        reading["light"] = means["light"] + RNG.normal(0, stds["light"])
        return reading

    if label == "SMOKE_RISK":
        reading["gas"] = means["gas"] + RNG.uniform(3.0, 6.0) * max(stds["gas"], 10.0)
        reading["light"] = means["light"] - RNG.uniform(2.6, 5.5) * max(stds["light"], 10.0)
        reading["humidity"] = means["humidity"] + RNG.uniform(2.0, 4.0) * stds["humidity"]
        reading["temperature"] = means["temperature"] + RNG.uniform(-0.5, 1.5) * stds["temperature"]
        reading["pressure"] = means["pressure"] + RNG.normal(0, max(stds["pressure"], 0.5))
        return reading

    if label == "WATER_LEAK_RISK":
        reading["humidity"] = means["humidity"] + RNG.uniform(3.0, 6.0) * stds["humidity"]
        reading["pressure"] = means["pressure"] + RNG.uniform(1.8, 4.0) * max(stds["pressure"], 0.5)
        reading["gas"] = means["gas"] + RNG.uniform(-0.8, 0.8) * max(stds["gas"], 10.0)
        reading["temperature"] = means["temperature"] + RNG.uniform(-0.8, 0.8) * stds["temperature"]
        reading["light"] = means["light"] + RNG.normal(0, stds["light"])
        return reading

    raise ValueError(f"Unknown label: {label}")


def make_dataset(samples_per_class: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []

    for label in CLASS_LABELS:
        for _ in range(samples_per_class):
            history = make_baseline()
            reading = latest_from_baseline(history)
            reading = inject(label, reading, history)
            features = build_deviation_features(reading, history)
            rows.append([features[name] for name in FEATURE_COLUMNS])
            labels.append(label)

    return np.asarray(rows, dtype=float), np.asarray(labels)


def main() -> None:
    x, y = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("Classes:", ", ".join(CLASS_LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=CLASS_LABELS))
    print()
    print(classification_report(y_test, predictions, labels=CLASS_LABELS))

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": CLASS_LABELS,
        "model_classes": tuple(model.classes_),
        "description": "Single Random Forest trained on deviation features for fire, smoke, and water leak risk.",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model to {Path(MODEL_PATH).resolve()}")


if __name__ == "__main__":
    main()
