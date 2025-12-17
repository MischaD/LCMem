import argparse
import os
import pandas as pd
import sys
sys.path.append("src/privacy/dataset")
from SiameseFPIDataset import SiameseNCEDataset  # pyright: ignore[reportMissingImports]
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any
import json
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import detectors from detector.py
from detector import (
    BaseMemorizationDetector,
    SiameseDetector,
    DenseClassifierDetector,
    NCCDetector,
    DarSiameseDetector,
    MemLDM2DDetector,
    MeanSquaredErrorDetector,
    SSIMDetector,
    Dinov3Detector,
    BeyondFIDDetector,
    ImageSpaceSiameseDetector,
    Dinov3PrecomputedDetector,
    DinoBackboneSiamese,
)


class BenchmarkEvaluator:
    """Evaluates methods using AUC, Precision, and Recall at a data-driven threshold."""

    def __init__(self):
        self.all_predictions: Dict[str, List[float]] = {}
        self.all_labels: Dict[str, List[int]] = {}
        self.all_metrics: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _at_target_recall(y_true, y_score, target_recall=0.95):
        """
        Return metrics at the smallest threshold achieving recall >= target_recall.
        Falls back to the max achievable recall if target not reachable.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # PR curve gives precision, recall for descending thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        # thresholds has length len(precisions) - 1 and corresponds to points [1..end]
        # We search over recalls[1:], thresholds[:], precisions[1:]
        idxs = np.where(recalls >= target_recall)[0]
        #if len(idxs) == 0:
        #    # not reachable: pick the highest recall point
        #    best_i = np.argmax(recalls[1:])
        #else:
        #    # choose first threshold that crosses target recall
        #    best_i = idxs[0]
        best_i = idxs[-1]

        thr = float(thresholds[best_i])
        # binarize at that threshold
        y_pred = (y_score >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = float(tp / (tp + fp + 1e-12))
        recall = float(tp / (tp + fn + 1e-12))
        fpr = float(fp / (fp + tn + 1e-12))

        return {
            "precision_at_recall": precision,
            "recall_target": float(target_recall),
            "recall_achieved": recall,
            "threshold_at_recall": thr,
            "fpr_at_recall": fpr,
        }

    @staticmethod
    def _pr_at_best_f1(y_true: List[int], y_score: List[float]) -> Tuple[float, float, float]:
        """
        Compute precision, recall at the threshold that maximizes F1 on the given scores.
        Returns (precision, recall, threshold).
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns an extra point for threshold list length + 1
        # Compute F1 for all valid points
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
        best_idx = np.argmax(f1s)
        # Map best_idx to threshold. If best_idx at last point, back off by one.
        if best_idx == len(thresholds):
            best_idx = len(thresholds) - 1
        best_thr = thresholds[max(0, best_idx)]
        # Recompute precision and recall at that threshold using binary predictions
        y_pred = (np.array(y_score) >= best_thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        return float(p), float(r), float(best_thr)

    def evaluate(self, predictions, true_labels, method_name, target_recall=0.95):
        self.all_predictions[method_name] = predictions
        self.all_labels[method_name] = true_labels

        metrics = {}

        # AUC is threshold free
        try:
            metrics["auc"] = float(roc_auc_score(true_labels, predictions))
        except ValueError:
            metrics["auc"] = 0.0

        # best-F1 operating point (you already had this logic)
        p_f1, r_f1, thr_f1 = self._pr_at_best_f1(true_labels, predictions)
        metrics.update({
            "precision_bestF1": p_f1,
            "recall_bestF1": r_f1,
            "threshold_bestF1": thr_f1,
        })

        # fixed recall operating point
        m_at_r = self._at_target_recall(true_labels, predictions, target_recall=target_recall)
        metrics.update({
            "precision_atR": m_at_r["precision_at_recall"],
            "recall_target": m_at_r["recall_target"],
            "recall_achieved_atR": m_at_r["recall_achieved"],
            "threshold_atR": m_at_r["threshold_at_recall"],
            "fpr_atR": m_at_r["fpr_at_recall"],
        })

        self.all_metrics[method_name] = metrics
        return metrics

    def print_results(self, metrics: Dict[str, float], method_name: str):
        print(f"\n=== Results for {method_name} ===")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Precision@bestF1: {metrics['precision_bestF1']:.4f}")
        print(f"Recall@bestF1:    {metrics['recall_bestF1']:.4f}")
        print(f"Best threshold:   {metrics['threshold_bestF1']:.4f}")

    def plot_roc_curves(self, output_dir: str):
        plt.figure(figsize=(10, 8))
        plt.style.use('seaborn-v0_8')

        for method_name, predictions in self.all_predictions.items():
            labels = self.all_labels[method_name]
            fpr, tpr, _ = roc_curve(labels, predictions)
            auc_val = roc_auc_score(labels, predictions)
            plt.plot(fpr, tpr, linewidth=2, label=f'{method_name} (AUC = {auc_val:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Memorization Detection Methods', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plot_path = os.path.join(output_dir, "roc_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves plot saved to: {plot_path}")

    def _bar_plot(self, series: Dict[str, float], ylabel: str, title: str, outfile: str, ylimit: Tuple[float, float] = (0.0, 1.05)):
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn-v0_8')

        methods = list(series.keys())
        scores = list(series.values())

        bars = plt.bar(methods, scores, edgecolor='black', alpha=0.8)
        for bar, score in zip(bars, scores):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Detection Methods', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylim(*ylimit)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{title} saved to: {outfile}")

    def plot_auc_bars(self, output_dir: str):
        aucs = {m: v["auc"] for m, v in self.all_metrics.items()}
        self._bar_plot(
            aucs,
            ylabel="AUC",
            title="AUC Comparison Across Methods",
            outfile=os.path.join(output_dir, "auc_comparison.png"),
            ylimit=(0.0, 1.05),
        )

    def plot_precision_bars(self, output_dir: str):
        precs = {m: v["precision_bestF1"] for m, v in self.all_metrics.items()}
        self._bar_plot(
            precs,
            ylabel="Precision@bestF1",
            title="Precision Comparison Across Methods",
            outfile=os.path.join(output_dir, "precision_comparison.png"),
            ylimit=(0.0, 1.05),
        )

    def plot_recall_bars(self, output_dir: str):
        recs = {m: v["recall_bestF1"] for m, v in self.all_metrics.items()}
        self._bar_plot(
            recs,
            ylabel="Recall@bestF1",
            title="Recall Comparison Across Methods",
            outfile=os.path.join(output_dir, "recall_comparison.png"),
            ylimit=(0.0, 1.05),
        )

    def plot_precision_atR_bars(self, output_dir: str):
        data = {m: v["precision_atR"] for m, v in self.all_metrics.items()}
        self._bar_plot(
            data,
            ylabel=f"Precision @ Recall={next(iter(self.all_metrics.values()))['recall_target']:.2f}",
            title="Precision at High Recall",
            outfile=os.path.join(output_dir, "precision_atR.png"),
            ylimit=(0.0, 1.05),
        )
    def save_results(self, output_dir: str, debug: bool, dataset_size: int, total_pairs: int):
        os.makedirs(output_dir, exist_ok=True)

        # Save numerical results as JSON
        output_path = os.path.join(output_dir, "benchmark_results.json")
        payload = {
            "results": self.all_metrics,
            "_debug": debug,
            "_dataset_size": dataset_size,
            "_total_pairs": total_pairs,
        }
        
        if os.path.exists(output_path):
            # Load existing results and append new ones
            with open(output_path, 'r') as f:
                existing = json.load(f)
            existing["results"].update(self.all_metrics)
            payload = existing
            
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Also save as CSV for quick tabular use
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        csv_rows = []
        for m, vals in self.all_metrics.items():
            row = {"method": m}
            row.update(vals)
            csv_rows.append(row)
        df = pd.DataFrame(csv_rows).sort_values("method")
        
        if os.path.exists(csv_path):
            # Load existing CSV and append new rows
            existing_df = pd.read_csv(csv_path)
            # Remove any existing rows for the same methods we're adding
            existing_df = existing_df[~existing_df["method"].isin(df["method"])]
            df = pd.concat([existing_df, df]).sort_values("method")
            
        df.to_csv(csv_path, index=False)

        # Plots
        self.plot_roc_curves(output_dir)
        self.plot_auc_bars(output_dir)
        self.plot_precision_bars(output_dir)
        self.plot_recall_bars(output_dir)
        self.plot_precision_atR_bars(output_dir)


def create_transforms(input_size: int = 512):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])


def run_benchmark(csv_path: str, basedir: str, output_dir: str, input_size: int = 512, seed: int = 42, debug: bool = False, target_recall: float = 0.95, ds_name="all", islatent=False, batch_size: int=32, fitmode=False):
    print("Starting benchmark with:")
    print(f"CSV file: {csv_path}")
    print(f"Base directory: {basedir}")
    print(f"Output directory: {output_dir}")
    print(f"Input size: {input_size}")
    print(f"ds name: {ds_name}")
    print(f"Random seed: {seed}")
    print(f"Is Latent: {islatent}")
    print(f"Debug mode: {debug}")

    if islatent: 
        transform = lambda x: x
    else: 
        transform = create_transforms(input_size)

    if ds_name != "all": 
        tmp_df = pd.read_csv(csv_path)
        tmp_df = tmp_df[tmp_df.ds_name == ds_name]
        tmp_df.to_csv(".tmp.csv")
        csv_path = ".tmp.csv"
        output_dir = os.path.join(output_dir , ds_name)

    print("\nCreating test dataset...")
    # Use 4 channels for latent space, 3 for images
    n_channels = 4 if islatent else 3

    if fitmode: 
        phase = "TRAIN"
        print("Fitmode activated. Computing metrics on train data to fit the models")
    else: 
        phase = "TEST"

    test_dataset = SiameseNCEDataset(phase=phase, n_channels=n_channels, transform=transform, filelist=csv_path, basedir=basedir, seed=seed, is_latent=islatent, fitmode=fitmode)
    print(f"Found n={len(test_dataset)} images")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    # Alias for backward compatibility
    PackhausSiameseDetector = SiameseDetector

    mse = MeanSquaredErrorDetector()
    mse.name = "MSE"
        
    # 2. SSIM
    ssim = SSIMDetector()
    ssim.name = "SSIM"
        
    # 3. NCC
    ncc = NCCDetector()
    ncc.name = "NCC"
        
    # 4. Random [17]
    random_det = BeyondFIDDetector("random")
    random_det.name = "Rnd-Weights"
            
        
    # ========== FOUNDATION MODELS ==========
    # 6. BYOL [8]
    byol = BeyondFIDDetector("byol")
    byol.name = "BYOL"
        
    # 7. CLIP [20]
    clip = BeyondFIDDetector("clip")
    clip.name = "CLIP"
        
    # 8. ConvNeXt [16]
    convnext = BeyondFIDDetector("convnext")
    convnext.name = "ConvNeXt"
        
    # 9. data2vec [2]
    data2vec = BeyondFIDDetector("data2vec")
    data2vec.name = "data2vec"
        
    # 10. Inception [24]
    inception = BeyondFIDDetector("inception")
    inception.name = "Inception"
        
    # 11. MAE [10]
    mae = BeyondFIDDetector("mae")
    mae.name = "MAE"
        
    # 12. SwAV [4]
    swav = BeyondFIDDetector("swav")
    swav.name = "SwAV"
        
    # 13. DINOv2 [18]
    dinov2 = BeyondFIDDetector("dinov2")
    dinov2.name = "DINOv2"
        
    # 14. DINOv3 [1]
    dinov3 = Dinov3Detector(name="DINOv3", layer=1, image_size=512)
    dinov3.name = "DINOv3"
        

    siam_ntxent_unsup = DarSiameseDetector(
    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/CameraReady_Dar_Unsuper/CameraReady_Dar_Unsuper_best_network.pth",
    packhaus_in_channels=3,
    packhaus_n_features=1024,
    image_size=512,
    network="ResNet-101"
    )
    siam_ntxent_unsup.name = "Siamese + NT-Xent U"

    siam_ntxent_task = DarSiameseDetector(
    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/CameraReady_Dar/CameraReady_Dar_best_network.pth",
    packhaus_in_channels=3,
    packhaus_n_features=1024,
    image_size=512,
    network="ResNet-101"
    )
    siam_ntxent_task.name = "Siamese + NT-Xent"

    siam = PackhausSiameseDetector(
    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/CameraReady_Baseline/CameraReady_Baseline_best_network.pth",
    packhaus_in_channels=3,
    packhaus_n_features=1024,
    image_size=512,
    network="ResNet-101"
    )
    siam.name = "Siamese"

    # ========== Ours ==========
    # Use ImageSpaceSiameseDetector to compute latents on-the-fly from images
    ours = ImageSpaceSiameseDetector(
    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/models/camera_ready08_wnorm_best_network.pth",
    vae_model_path="stabilityai/stable-diffusion-2",  # VAE model path for encoding
    packhaus_in_channels=4,
    packhaus_n_features=512,
    image_size=64,  # Size to resize latents to before passing to Siamese network
    latent_size=64,  # Expected latent size from VAE
    network='ConvNeXt-Tiny',
    final_mean=0.0,
    final_std=0.5,
    )
    ours.name = "Ours" # convnext tiny with alpha = 0.8 

    ours_ft_low = ImageSpaceSiameseDetector(
    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/FineTune-ImageSpace-stronger4/FineTune-ImageSpace-stronger4_best_network.pth",
    vae_model_path="stabilityai/stable-diffusion-2",  # VAE model path for encoding
    packhaus_in_channels=4,
    packhaus_n_features=512,
    image_size=64,  # Size to resize latents to before passing to Siamese network
    latent_size=64,  # Expected latent size from VAE
    network='ConvNeXt-Tiny',
    final_mean=0.0,
    final_std=0.5,
    )
    ours_ft_low.name = "Ours Two Stage" # convnext tiny with alpha = 0.8 


    #dino_vit7b = Dinov3Detector(name="DinoVitSPool", layer=1, image_size=512)

    #siam = SiameseDetector(
    #        packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/isambard/clm_highnoise_best_network.pth",
    #        packhaus_in_channels=4,
    #        packhaus_n_features=512,
    #        image_size=64, 
    #        network='ResNet-101',
    #)
    #siam.name = "ours_82" # ResNet no alpha 
    #siam = SiameseDetector(
    #        packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/models/clm_lownoise_alpha05_best_network.pth",
    #        packhaus_in_channels=4,
    #        packhaus_n_features=512,
    #        image_size=64, 
    #        network='ResNet-101',
    #)
    #siam.name = "outs_latent_alpha_resnet" Resnet101 latent with alpha = 0.5

    #siam = SiameseDetector(
    #        packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/models/camera_ready08_wnorm_best_network.pth",
    #        packhaus_in_channels=4,
    #        packhaus_n_features=512,
    #        image_size=64, 
    #        network='ConvNeXt-Tiny',
    #)
    #siam.name = "ours_CNT_latent_alpha_wnorm" # convnext tiny with alpha = 0.8 
    siam_twostage = SiameseDetector(
            packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/FineTune-ImageSpace-stronger4/FineTune-ImageSpace-stronger4_best_network.pth",
            packhaus_in_channels=4,
            packhaus_n_features=512,
            image_size=64, 
            network='ConvNeXt-Tiny',
    )
    siam_twostage.name = "siam_twostage_new" # convnext tiny with alpha = 0.8 


    #siam = SiameseDetector(
    #        packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/isambard/clm_convnext_convnext-tiny_best_network.pth",
    #        packhaus_in_channels=4,
    #        packhaus_n_features=512,
    #        image_size=64, 
    #        network='ConvNeXt-Tiny',
    #)
    #siam.name = "ours_alphaconvnext"






    #pack_repr = SiameseDetector(
    #        packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/config_baseline_singlegpu/config_baseline_singlegpu_best_network.pth",
    #        packhaus_in_channels=3,
    #        packhaus_n_features=128,
    #        image_size=512, 
    #        network='ResNet-101',
    #)
    #pack_repr.name = "packhaus_baseline"



    #dar_uncon = DarSiameseDetector(
    #    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/CameraReady_Dar_Unsuper/CameraReady_Dar_Unsuper_best_network.pth",
    #    packhaus_in_channels=3,
    #    packhaus_n_features=1024,
    #    image_size=512, 
    #    network='ResNet-101',
    #)
    #dar_uncon.name = "dar_uncond_baseline"

    #dar_cond = DarSiameseDetector(
    #    packhaus_model_path="/vol/ideadata/ed52egek/pycharm/syneverything/privacy/fpi_mem/archive/CameraReady_Dar/CameraReady_Dar_best_network.pth",
    #    packhaus_in_channels=3,
    #    packhaus_n_features=1024,
    #    image_size=512, 
    #    network='ResNet-101',
    #)
    #dar_cond.name = "dar_cond_baseline"

    #siam_downstream = DenseClassifierDetector(dataset_name=ds_name, )
    
    detectors = [  
            ours_ft_low ,
            mse, 
            ssim,
            ncc,
            random_det,
            byol,
            clip,
            convnext,
            data2vec,
            inception,
            mae,
            swav,
            dinov2,
            dinov3,
            siam_ntxent_unsup, 
            siam_ntxent_task,
            siam,     # ========== Ours ==========
            ours,     
    ]
    print(detectors)

    evaluator = BenchmarkEvaluator()
    os.makedirs(output_dir, exist_ok=True)

    for detector in detectors:
        print("\n" + "=" * 50)
        print(f"Evaluating {detector.name}")
        print("=" * 50)

        if fitmode: 
            print("Predicting on train set...")
        else: 
            print("Predicting on test set...")

        test_predictions, test_labels = detector.predict_batch(test_loader, show_progress=True)
        base_datasets = test_dataset.base_dataset

        # Get unique dataset names
        unique_datasets = list(set(base_datasets))

        # Evaluate overall performance (micro AUC)
        metrics = evaluator.evaluate(test_predictions, test_labels, detector.name, target_recall=target_recall)
        evaluator.print_results(metrics, f"{detector.name} (Test Overall - Micro AUC)")

        # Evaluate per dataset
        for dataset in unique_datasets:
            # Get indices for this dataset
            dataset_indices = [i for i, x in enumerate(base_datasets) if x == dataset]
            
            # Get predictions and labels for this dataset
            dataset_predictions = np.array(test_predictions)[dataset_indices]
            dataset_labels = np.array(test_labels)[dataset_indices]

            # Evaluate metrics for this dataset
            dataset_metrics = evaluator.evaluate(
                dataset_predictions, 
                dataset_labels,
                f"{detector.name}_{dataset}", 
                target_recall=target_recall
            )
            evaluator.print_results(dataset_metrics, f"{detector.name} (Test {dataset})")

    evaluator.save_results(
        output_dir=output_dir,
        debug=debug,
        dataset_size=len(test_dataset),
        total_pairs=len(test_dataset.pairs),
    )

    print("\n" + "=" * 50)
    print("Benchmark completed!")
    print("=" * 50)

    print("\nSummary of Results:")
    if fitmode: 
        print("\n!!!! Computed on train to get thresholds!!!")

    for method, vals in evaluator.all_metrics.items():
        print(f"{method}: AUC={vals['auc']:.4f}  RecallatBestF1={vals['recall_bestF1']:.4f}  PrecisionAtHighRecall={vals['precision_atR']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memorization detection methods using AUC, Precision, and Recall")
    parser.add_argument("--filelist", type=str, required=True, help="CSV file with columns 'Split', 'id', and 'path'")
    parser.add_argument("--basedir", type=str, required=True, help="Base directory for image paths")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--input_size", type=int, default=512, help="Size to resize images to")
    parser.add_argument("--batch_size", type=int, default=32, help="Size to resize images to")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--fitmode", action="store_true", help="Enable to start fitmode, which only computes the best and the high recall thresholds")
    parser.add_argument("--islatent", action="store_true", help="Uselatent")
    parser.add_argument("--target_recall", type=float, default=0.99, help="Target recall for operating point")
    parser.add_argument("--ds_name", type=str, choices=['celeba', 'ctrate', 'mimic', 'cxr-lt', 'isic', 'all'], default='all', help="Dataset name to evaluate")


    args = parser.parse_args()

    if not os.path.exists(args.filelist):
        raise FileNotFoundError(f"CSV file not found: {args.filelist}")
    if not os.path.exists(args.basedir):
        raise FileNotFoundError(f"Base directory not found: {args.basedir}")

    run_benchmark(args.filelist, args.basedir, args.output_dir, args.input_size, args.seed, args.debug, target_recall=args.target_recall, ds_name=args.ds_name, islatent=args.islatent, batch_size=args.batch_size, fitmode=args.fitmode)


if __name__ == "__main__":
    main()