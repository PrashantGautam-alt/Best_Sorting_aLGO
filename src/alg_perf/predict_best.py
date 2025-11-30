"""
This is smoother CLI + default behavior version.

If run WITHOUT any arguments:
    automatically loads model.pkl
    runs a random demo array of size 1000
"""

import argparse
import json
import time
import statistics
import joblib
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from features import extract_features
from sorters import SORTERS


EXPECTED_COLS = [
    "length", "mean", "std", "min_val", "max_val",
    "range", "fraction_unique", "duplicates_ratio",
    "sortedness_score", "runs_count", "runs_fraction"
]


# Utility loaders and helpers
def load_model(path: str = "model.pkl") -> Any:
    try:
        model = joblib.load(path)
        print(f"Loaded model from {path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {path}. Train model first.")
        return None


def _time_fn(fn, arr, repeats: int = 3) -> float:
    """Return median time in seconds."""
    times = []
    for _ in range(repeats):
        a = list(arr)
        t0 = time.perf_counter()
        fn(a)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(statistics.median(times))


def prepare_feature_df(feats: Dict[str, Any], model: Any) -> pd.DataFrame:
    df = pd.DataFrame([feats])
    df_num = df.select_dtypes(include=[np.number])

    if isinstance(model, dict) and model.get("feature_names"):
        cols = [c for c in model["feature_names"] if c in df_num.columns]
        if cols:
            return df_num[cols]

    if all(col in df_num.columns for col in EXPECTED_COLS):
        return df_num[EXPECTED_COLS]

    return df_num  


def map_prediction_to_algo(pred, model: Any) -> Optional[str]:
    if isinstance(model, dict) and model.get("algo_cols"):
        algo_cols = model["algo_cols"]
        try:
            idx = int(pred)
            if 0 <= idx < len(algo_cols):
                name = algo_cols[idx]
                if name.startswith("time_"):
                    return name.replace("time_", "")
                elif name.startswith("algo_") and name.endswith("_time"):
                    return name[len("algo_"):-len("_time")]
                else:
                    return name
        except Exception:
            pass

    if isinstance(pred, str):
        return pred

    try:
        return str(pred)
    except Exception:
        return None


# Core prediction logic

def predict_for_list(
    arr: List[int],
    model: Any,
    confidence_threshold: float = 0.0,
    measure_predicted: bool = True,
    repeats: int = 3
) -> Dict[str, Any]:

    out = {}

    # Feature extraction
    t0 = time.perf_counter()
    feats = extract_features(arr)
    feature_time = time.perf_counter() - t0
    out["features"] = feats
    out["feature_time"] = feature_time

    df_feat = prepare_feature_df(feats, model)

    clf = model["clf"] if isinstance(model, dict) and "clf" in model else model
    rgrs = model.get("rgrs") if isinstance(model, dict) else None

    # Classifier inference
    t0 = time.perf_counter()
    pred = clf.predict(df_feat)[0]
    model_infer_time = time.perf_counter() - t0
    out["model_infer_time"] = model_infer_time

    # Probabilities
    probs = None
    if hasattr(clf, "predict_proba"):
        try:
            probs_arr = clf.predict_proba(df_feat)[0]
            classes = clf.classes_
            probs = dict(zip(map(str, classes), probs_arr.tolist()))
            out["probs"] = probs
        except Exception:
            probs = None

    # Confidence fallback
    if confidence_threshold and probs is not None:
        chosen_prob = None
        for cls_v, p in zip(clf.classes_, probs_arr):
            if str(cls_v) == str(pred):
                chosen_prob = float(p)
                break
        if chosen_prob is not None and chosen_prob < confidence_threshold:
            out["confidence_fallback"] = {
                "chosen_prob": chosen_prob,
                "threshold": confidence_threshold,
                "action": "fallback to builtin"
            }
            pred = "builtin"

    out["raw_prediction"] = pred

    # Regressor estimates (if available)
    predicted_est_time = None
    if rgrs:
        t0 = time.perf_counter()
        est_times = [float(r.predict(df_feat)[0]) for r in rgrs]
        reg_time = time.perf_counter() - t0
        out["regressor_infer_time"] = reg_time
        out["model_infer_time"] += reg_time
        out["estimated_times"] = est_times

        try:
            idx = int(pred)
            predicted_est_time = est_times[idx]
        except:
            predicted_est_time = min(est_times)
        out["predicted_est_time"] = predicted_est_time

    # Baseline builtin sort
    baseline_time = _time_fn(SORTERS["builtin"], arr, repeats=repeats)
    out["baseline_time"] = baseline_time

    overhead = feature_time + out["model_infer_time"]
    out["overhead"] = overhead

    # Predicted algorithm measurement
    pred_key = map_prediction_to_algo(pred, model)
    predicted_measured_time = None
    if measure_predicted and pred_key in SORTERS:
        predicted_measured_time = _time_fn(SORTERS[pred_key], arr, repeats=repeats)
        out["predicted_measured_time"] = predicted_measured_time
        out["net_saved_measured"] = baseline_time - (predicted_measured_time + overhead)
    else:
        out["predicted_measured_time"] = None
        out["net_saved_measured"] = None

    if predicted_est_time is not None:
        out["net_saved_est"] = baseline_time - (predicted_est_time + overhead)

    out["predicted_algo_key"] = pred_key

    return out


# CLI + DEFAULT behavior

def main_cli():
    parser = argparse.ArgumentParser(description="Predict best sorting algorithm for an array")
    parser.add_argument("--model", "-m", help="Path to model.pkl")
    parser.add_argument("--demo", choices=["nearly_sorted", "reverse", "random", "small_random"])
    parser.add_argument("--random", type=int)
    parser.add_argument("--measure-pred", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--out-json", type=str)
    parser.add_argument("--file", type=str)
    args, unknown = parser.parse_known_args()

    if len(sys.argv) == 1:
        print("No arguments provided — running default demo...")
        args.model = args.model or "model.pkl"
        args.demo = "random"
        args.random = 1000  # size
        args.measure_pred = False

    # Load model
    model = load_model(args.model)
    if model is None:
        sys.exit(1)

    results = []

    # Demo cases
    if args.demo:
        if args.demo == "nearly_sorted":
            arr = list(range(2000))
            arr[50], arr[1800] = arr[1800], arr[50]
        elif args.demo == "reverse":
            arr = list(range(2000, 0, -1))
        elif args.demo == "random":
            arr = [random.randint(0, 10**6) for _ in range(args.random or 1000)]
        elif args.demo == "small_random":
            arr = [random.randint(0, 100) for _ in range(200)]
        else:
            raise ValueError("Unknown demo type")

        res = predict_for_list(arr, model, confidence_threshold=args.confidence_threshold, measure_predicted=args.measure_pred, repeats=args.repeats)
        pretty_print(res)
        results.append(res)

    # JSON batch input
    elif args.file:
        with open(args.file) as f:
            arrays = json.load(f)
        for arr in arrays:
            res = predict_for_list(arr, model, confidence_threshold=args.confidence_threshold, measure_predicted=args.measure_pred, repeats=args.repeats)
            pretty_print(res)
            results.append(res)

    # Random array by size
    elif args.random:
        arr = [random.randint(0, 10**6) for _ in range(args.random)]
        res = predict_for_list(arr, model, confidence_threshold=args.confidence_threshold, measure_predicted=args.measure_pred, repeats=args.repeats)
        pretty_print(res)
        results.append(res)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print("Saved JSON to", args.out_json)


def pretty_print(res: Dict[str, Any]):
    print("\n===== Prediction Summary =====")
    print(f"Predicted algorithm: {res.get('predicted_algo_key')}")
    print(f"Feature time: {res['feature_time']:.6f}s")
    print(f"Model time: {res['model_infer_time']:.6f}s")
    print(f"Overhead: {res['overhead']:.6f}s")
    print(f"Baseline time: {res['baseline_time']:.6f}s")
    if res.get("predicted_est_time") is not None:
        print(f"Predicted est: {res['predicted_est_time']:.6f}s")
        print(f"Net est saved: {res['net_saved_est']:.6f}s")
    if res.get("predicted_measured_time") is not None:
        print(f"Predicted measured: {res['predicted_measured_time']:.6f}s")
        print(f"Net measured saved: {res['net_saved_measured']:.6f}s")
    if res.get("confidence_fallback"):
        print("Fallback:", res["confidence_fallback"])
    print("==============================\n")


if __name__ == "__main__":
    main_cli()
