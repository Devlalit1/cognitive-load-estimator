"""Training and inference for the three-class cognitive-load model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

try:
    from .window_buffer import FEATURE_NAMES
except ImportError:
    from window_buffer import FEATURE_NAMES

LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLORS = {0: "#1a9850", 1: "#fdae61", 2: "#d73027"}
MODEL_PATH = Path(__file__).parent.parent / "data" / "trained_model.pkl"


class CognitiveLoadModel:
    """Small wrapper around a scaled Random Forest classifier."""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        classes: Iterable[str] = ("Low", "Medium", "High"),
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.classes = tuple(classes)

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "CognitiveLoadModel":
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or not {"model", "scaler"} <= bundle.keys():
            raise ValueError(f"Invalid model bundle: {path}")
        return cls(
            bundle["model"],
            bundle["scaler"],
            bundle.get("classes", ("Low", "Medium", "High")),
        )

    def save(self, path: str | Path = MODEL_PATH) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "classes": self.classes},
            target,
        )

    def predict(self, vector: np.ndarray) -> tuple[int, dict[str, float]]:
        """Return numeric class id and class probabilities for one vector."""
        values = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        expected = getattr(self.scaler, "n_features_in_", values.shape[1])
        if values.shape[1] != expected:
            raise ValueError(f"Expected {expected} features, received {values.shape[1]}")
        scaled = self.scaler.transform(values)
        prediction = int(self.model.predict(scaled)[0])
        probabilities = self.model.predict_proba(scaled)[0]
        classes = getattr(self.model, "classes_", range(len(probabilities)))
        return prediction, {
            self.classes[int(class_id)]: float(probability)
            for class_id, probability in zip(classes, probabilities)
        }


def _session_rows(data_dir: Path) -> Any:
    import pandas as pd

    files = sorted((data_dir / "sessions").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No session CSVs found in {data_dir / 'sessions'}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        label_file = data_dir / "labels" / f"{path.stem}_label.txt"
        if "load_label" not in frame and label_file.exists():
            frame["load_label"] = label_file.read_text(encoding="utf-8").strip().title()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def train(data_dir: str | Path = MODEL_PATH.parent) -> CognitiveLoadModel:
    """Train a model from session CSVs and persist it to ``MODEL_PATH``."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    df = _session_rows(Path(data_dir))
    missing = [column for column in FEATURE_NAMES if column not in df]
    if missing:
        raise ValueError(f"Session data is missing feature columns: {missing}")
    if "load_label" not in df:
        raise ValueError("Session data must contain a load_label column")

    clean = df.dropna(subset=FEATURE_NAMES + ["load_label"])
    labels = clean["load_label"].astype(str).str.title()
    valid = labels.isin(LABEL_NAMES.values())
    clean = clean.loc[valid]
    labels = labels.loc[valid]
    if len(clean) < 10 or labels.nunique() < 2:
        raise ValueError("Need at least 10 rows and two load classes to train")

    scaler = StandardScaler()
    x = scaler.fit_transform(clean[FEATURE_NAMES].to_numpy(dtype=np.float32))
    y = labels.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()
    classifier = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    classifier.fit(x, y)
    trained = CognitiveLoadModel(classifier, scaler)
    trained.save(MODEL_PATH)
    return trained


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    else:
        print("Use --train after collecting labeled session CSV files.")
