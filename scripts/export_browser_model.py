from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model-v1.pkl"
OUTPUT_PATH = ROOT / "deployment_web" / "frontend" / "public" / "model" / "browser-model.json"


def rounded(value: Any, decimals: int = 6) -> Any:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        return np.round(array.astype(float), decimals).tolist()
    return array.tolist()


def export_tree(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    return {
        "childrenLeft": rounded(tree.children_left),
        "childrenRight": rounded(tree.children_right),
        "feature": rounded(tree.feature),
        "threshold": rounded(tree.threshold, 6),
        "value": rounded(tree.value[:, 0, :], 6),
    }


def main() -> None:
    with MODEL_PATH.open("rb") as file:
        package = pickle.load(file)

    scaler = package["scaler"].named_steps["standard_scaler"]
    random_forest = package["model"].named_estimators_["random_forest"]
    knn = package["model"].named_estimators_["knn"]

    payload = {
        "version": 1,
        "modelName": f"{package.get('model_name', 'SER model')} - Browser RandomForest",
        "suite": package.get("suite", "Unknown"),
        "featureSet": package.get("feature_set", "Unknown"),
        "classNames": package["class_names"],
        "featureNames": package["all_feature_names"],
        "standardScaler": {
            "mean": rounded(scaler.mean_, 6),
            "scale": rounded(scaler.scale_, 6),
        },
        "randomForest": {
            "nClasses": int(random_forest.n_classes_),
            "nTrees": len(random_forest.estimators_),
            "trees": [export_tree(tree) for tree in random_forest.estimators_],
        },
        "knn": {
            "nClasses": len(package["class_names"]),
            "nNeighbors": int(knn.n_neighbors),
            "x": rounded(knn._fit_X, 6),
            "y": rounded(knn._y),
        },
        "testMetrics": package.get("test_metrics", {}),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
