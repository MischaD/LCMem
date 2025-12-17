#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def find_runs(root):
    # e.g., .../archive/Siamese_ResNet50_dinol0/, .../Siamese_ResNet50_l2/, etc.
    pattern = os.path.join(root, "Siamese_ResNet50_*")
    return sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])

def layer_name_from_dir(path):
    base = os.path.basename(os.path.normpath(path))
    # Expect "Siamese_ResNet50_<layername>"
    parts = base.split("Siamese_ResNet50_", 1)
    return parts[1] if len(parts) == 2 else base

def load_labels_predictions(run_dir):
    fpath = os.path.join(run_dir, f"{os.path.basename(run_dir)}_labels_predictions.txt")
    if not os.path.exists(fpath):
        return None
    # File is tab-separated with a header: Label  Prediction  PredictionThresholded
    df = pd.read_csv(fpath, sep="\t")
    # Coerce to numeric
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")
    df = df.dropna(subset=["Label", "Prediction"])
    y_true = df["Label"].values
    y_score = df["Prediction"].values
    return y_true, y_score

def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves for all Siamese_ResNet50 layer evaluations.")
    parser.add_argument("--archive_root", required=True,
                        help="Path to archive root (e.g., /vol/.../privacy/packhaus/archive)")
    parser.add_argument("--out", default="siamese_layers_roc.png",
                        help="Output path for the ROC figure")
    args = parser.parse_args()

    runs = find_runs(args.archive_root)
    if not runs:
        print(f"No runs found in {args.archive_root}")
        return

    plt.figure(figsize=(9, 7))
    plotted = 0
    results = []

    for run_dir in runs:
        lp = load_labels_predictions(run_dir)
        if lp is None:
            continue
        y_true, y_score = lp
        if y_true.size == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        lname = layer_name_from_dir(run_dir)
        plt.plot(fpr, tpr, lw=2, label=f"{lname} (AUC={auc_val:.3f})")
        results.append((lname, auc_val, run_dir))
        plotted += 1

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves across Siamese_ResNet50 DINO layers")
    plt.legend(loc="lower right", fontsize=9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    if plotted == 0:
        print("Found no valid labels_predictions files to plot.")
        return

    # Print a concise AUC table
    results.sort(key=lambda x: x[1], reverse=True)
    print("AUC by layer:")
    for lname, auc_val, run_dir in results:
        print(f"- {lname}: {auc_val:.4f}  ({run_dir})")
    print(f"\nSaved ROC figure to: {args.out}")

if __name__ == "__main__":
    main()