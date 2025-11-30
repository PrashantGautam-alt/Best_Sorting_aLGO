
"""
Standalone plotting script.

Use it as :
  python3 plot_eval.py --model model.pkl --csv sorting_data.csv --out plots

What it does:
 Loads model (meta-dict or plain classifier)
 Loads CSV (expects time_* columns, e.g., time_quick/time_merge/time_heap)
 Infers feature columns (or uses model['feature_names'])
 Predicts labels for the CSV feature matrix
 Estimates model inference time (used as overhead)
 Produces:
     plots/confusion_matrix.png
     plots/feature_importance.png
     plots/net_savings_hist_cdf.png
     plots/net_savings_by_class.png
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
import time

plt.rcParams.update({'figure.max_open_warning': 0})


def plot_confusion_matrix(y_true, y_pred, labels, out_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2. if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path='feature_importance.png', topk=30):
    if isinstance(model, dict) and 'clf' in model:
        clf = model['clf']
    else:
        clf = model

    if not hasattr(clf, 'feature_importances_'):
        print("Model has no feature_importances_. Skipping feature importance plot.")
        return

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:topk]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(names))))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Feature importances (top {})".format(len(names)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    plt.close(fig)


def plot_net_savings_from_df(eval_df, out_dir):
    out_hist = os.path.join(out_dir, 'net_savings_hist_cdf.png')
    out_box = os.path.join(out_dir, 'net_savings_by_class.png')

    if 'net_saved_est' not in eval_df.columns and 'net_saved_actual' not in eval_df.columns:
        print("No net_saved columns found — skipping net-savings plots.")
        return

    col = 'net_saved_est' if 'net_saved_est' in eval_df.columns else 'net_saved_actual'
    data = eval_df[col].dropna().values
    if data.size == 0:
        print("No net-savings data (empty) — skipping.")
        return

    # histogram + CDF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data, bins=50)
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_title(f"Net time saved distribution ({col})")
    axes[0].set_xlabel("Net time saved (seconds)")
    axes[0].set_ylabel("Count")

    s = np.sort(data)
    cdf = np.arange(1, len(s)+1) / len(s)
    axes[1].plot(s, cdf)
    axes[1].set_title("CDF of net time saved")
    axes[1].set_xlabel("Net time saved (seconds)")
    axes[1].set_ylabel("Fraction of cases ≤ x")

    fig.tight_layout()
    fig.savefig(out_hist, dpi=200)
    print(f"Wrote {out_hist}")
    plt.close(fig)

    # per-class boxplot (predicted_label required)
    if 'predicted_label' in eval_df.columns:
        dfc = eval_df.dropna(subset=[col, 'predicted_label'])
        if not dfc.empty:
            groups = dfc.groupby('predicted_label')[col].apply(list)
            labels = list(groups.index)
            data = [groups[lbl] for lbl in labels]
            fig, ax = plt.subplots(figsize=(max(6, 1.2*len(labels)), 5))
            ax.boxplot(data, labels=labels, vert=True)
            ax.set_ylabel("Net time saved (seconds)")
            ax.set_title("Net savings by predicted class")
            fig.tight_layout()
            fig.savefig(out_box, dpi=200)
            print(f"Wrote {out_box}")
            plt.close(fig)
        else:
            print("No per-class data to plot boxplots.")
    else:
        print("predicted_label column not present — skipping per-class boxplot.")


def estimate_model_infer_time(model, X_sample, repeats=10):
    t0 = time.perf_counter()
    for _ in range(repeats):
        if isinstance(model, dict) and 'clf' in model:
            clf = model['clf']
            _ = clf.predict(X_sample)
            if 'rgrs' in model and model['rgrs'] is not None:
                for r in model['rgrs']:
                    _ = r.predict(X_sample)
        else:
            model.predict(X_sample)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main(model_path, csv_path, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading model:", model_path)
    model = joblib.load(model_path)

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    # Find algorithm timing columns (time_*)
    algo_cols = [c for c in df.columns if c.startswith('time_')]
    if not algo_cols:
        raise RuntimeError("No algorithm timing columns found (expect columns starting with 'time_').")

    # Baseline: choose min across available algo times per row
    df['baseline_time'] = df[algo_cols].min(axis=1)

    # Feature columns: numeric columns excluding meta & algo columns
    non_features = set(['pattern', 'seed', 'fastest_label']) | set(algo_cols) | {'baseline_time'}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in non_features]

    # Prefer model-saved feature names if present
    if isinstance(model, dict) and model.get('feature_names'):
        feature_cols = [c for c in model['feature_names'] if c in df.columns]

    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns detected. Check your CSV or model feature_names.")

    print("Using feature columns:", feature_cols[:10], " (total", len(feature_cols), ")")

    X = df[feature_cols].astype(float)

    # Predict labels
    if isinstance(model, dict) and 'clf' in model:
        clf = model['clf']
    else:
        clf = model
    print("Running classifier predictions on benchmark dataset...")
    preds = clf.predict(X)
    df['predicted_label'] = preds

    # Estimate model inference time using up to 50 rows
    sample_n = min(50, len(X))
    model_infer_time = estimate_model_infer_time(model, X.iloc[:sample_n], repeats=10)
    print(f"Estimated model inference time (per row): {model_infer_time:.6f} s")


    # Save a small eval_per_row-like CSV for inspection
    out_eval_csv = os.path.join(out_dir, 'eval_per_row_generated.csv')
    to_save_cols = ['predicted_label', 'baseline_time'] + (['predicted_est_time','net_saved_est'] if 'predicted_est_time' in df.columns else [])
    df[to_save_cols].to_csv(out_eval_csv, index=False)
    print("Wrote", out_eval_csv)

    #  Confusion matrix
    if 'fastest_label' in df.columns:
        try:
            labels = np.unique(np.concatenate([df['fastest_label'].values, df['predicted_label'].astype(str).values]))
            plot_confusion_matrix(df['fastest_label'].values, df['predicted_label'].astype(str).values, labels, out_path=os.path.join(out_dir, 'confusion_matrix.png'))
        except Exception as e:
            print("Could not create confusion matrix:", e)
    else:
        print("fastest_label not present in CSV — skipping confusion matrix.")

    #  Feature importance
    plot_feature_importance(model, feature_cols, out_path=os.path.join(out_dir, 'feature_importance.png'))

    #  Net savings plots (from generated per-row data)
    plot_net_savings_from_df(df, out_dir)

    # Save aggregate report
    report = {
        'num_rows': int(len(df)),
        'model_infer_time_sec': float(model_infer_time),
    }
    if 'net_saved_est' in df.columns:
        s = df['net_saved_est'].dropna()
        report['net_saved_est_mean'] = float(s.mean())
        report['net_saved_est_median'] = float(s.median())
        report['net_saved_est_pct_positive'] = float((s > 0).mean())
    report_path = os.path.join(out_dir, 'plot_eval_report.json')
    with open(report_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print("Wrote aggregate report:", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', required=True)
    parser.add_argument('--csv','-c', required=True)
    parser.add_argument('--out','-o', default='plots')
    args = parser.parse_args()
    main(args.model, args.csv, args.out)
