"""
Benchmark framework for memorization detection methods.
Each method inherits from a base class and implements predict function.
Uses only AUC ROC as evaluation metric, no training required.
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any
import json
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

# Import detectors from detector.py
from detector import (
    BaseMemorizationDetector,
    SiameseDetector,
    ImageSpaceSiameseDetector,
    NCCDetector,
    DarSiameseDetector,
    MemLDM2DDetector,
    MeanSquaredErrorDetector,
    SSIMDetector,
    Dinov3Detector,
    BeyondFIDDetector,
    Dinov3PrecomputedDetector,
    DinoBackboneSiamese,
)

# Alias for backward compatibility
PackhausSiameseDetector = SiameseDetector
# Enable LaTeX font rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})



class MemorizationDetectionDataset(Dataset):
    """Dataset for memorization detection that creates pairs of images."""
    
    def __init__(self, csv_path: str, base_dir: str, split: str = "test", transform=None, seed: int = 42, debug: bool = False, robustness_transform=None, max_images: int = None, sameimage: bool = False):
        """
        Args:
            csv_path: Path to CSV file with columns 'Split' and 'id'
            base_dir: Base directory for image paths
            split: Which split to use ('train', 'val', 'test')
            transform: Image transformations
            seed: Random seed for deterministic pair generation
            debug: If True, limit to 100 images for faster testing
            robustness_transform: robustness_transform to apply to the second image to check for robustness
        """
        self.base_dir = base_dir
        self.transform = transform
        self.debug = debug
        self.robustness_transform = robustness_transform
        self.max_images = max_images
        self.processor = None
        self.sameimage = sameimage
        
        # Set random seed for deterministic sampling
        np.random.seed(seed)
        
        # Load CSV and filter by split
        df = pd.read_csv(csv_path)
        df = df[df['Split'] == split].reset_index(drop=True)
        
        # Debug mode: limit to 100 images
        if debug:
            print(f"DEBUG MODE: Limiting dataset to 100 images")
            df = df.head(100)


        # Group by id to get all images for each identity
        self.id_groups = df.groupby('id')['path'].apply(list).to_dict()
        self.ids = list(self.id_groups.keys())
       
        # Create positive pairs (same id) and negative pairs (different ids)
        self.positive_pairs = []
        self.negative_pairs = []
        
        # Generate positive pairs (same id) - deterministic
        for id_name, paths in self.id_groups.items():
            if len(paths) >= 2:
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        self.positive_pairs.append((paths[i], paths[j], 1))
        
        # Generate negative pairs (different ids) - deterministic
        for i in range(len(self.positive_pairs)): 
            pos = df.iloc[i%len(df)]
            id1 = pos["id"].item()
            while True: 
                negative = df.sample(n=1) 
                if id1 != negative["id"].item(): 
                    break

            self.negative_pairs.append((pos["path"], negative["path"].item(), 0))
        
        # Combine all pairs and shuffle deterministically
        if self.sameimage: 
            # Use same image for both inputs (identity pairs)
            all_imgs = list(df.groupby('id').sample(1)['path']) 
            self.all_pairs = [(x, x, 1) for x in all_imgs]
        else:
            # Use all pairs (positive + negative) for fitting
            # For testing, we'll filter to only positives later if needed
            self.all_pairs = self.positive_pairs + self.negative_pairs

        np.random.shuffle(self.all_pairs)
        if self.max_images is not None:
            print(f"Limiting {split} dataset to first {self.max_images} pairs")
            self.all_pairs = self.all_pairs[:self.max_images]
        
        print(f"Split: {split}, Total pairs: {len(self.all_pairs)}")
        if debug:
            print(f"DEBUG: Limited to {len(df)} images, resulting in {len(self.all_pairs)} pairs")
    
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        path1, path2, label = self.all_pairs[idx]
        
        # Load images
        img1 = Image.open(os.path.join(self.base_dir, path1)).convert('RGB')
        img2 = Image.open(os.path.join(self.base_dir, path2)).convert('RGB')

        if self.sameimage: 
            label = 1
            img2 = img1.copy()

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else: 
            pass

        if self.robustness_transform: 
            img2 = self.robustness_transform(img2)

        # Clamp images to [0, 1] range to avoid PIL conversion issues
        # Some transforms (rotation, brightness, noise, etc.) can push values outside [0, 1]
        if isinstance(img1, torch.Tensor):
            img1 = torch.clamp(img1, 0.0, 1.0)
        if isinstance(img2, torch.Tensor):
            img2 = torch.clamp(img2, 0.0, 1.0)

        if self.processor: 
            img1 = TF.to_pil_image(img1)
            img2 = TF.to_pil_image(img2)
            inputs = self.processor(images=[img1, img2], return_tensors='pt')
            pv = inputs['pixel_values']
            img1, img2 = pv[0], pv[1]

        return img1, img2, label


class BenchmarkEvaluator:
    """Evaluates memorization detection methods using AUC ROC metric."""
    
    def __init__(self):
        self.metric_name = 'roc_auc'
        self.all_predictions = {}
        self.all_labels = {}
    
    def evaluate(self, predictions: List[float], true_labels: List[int], method_name: str) -> float:
        """
        Evaluate predictions using AUC ROC metric.
        
        Args:
            predictions: List of prediction scores
            true_labels: List of true labels (0 or 1)
            method_name: Name of the method for storing results
            
        Returns:
            AUC ROC score
        """
        # Store predictions and labels for plotting
        self.all_predictions[method_name] = predictions
        self.all_labels[method_name] = true_labels
        
        try:
            acc = accuracy_score(true_labels, predictions)
            rec = recall_score(true_labels, predictions)
            return acc, rec
        except ValueError:
            print("Warning: Could not compute AUC ROC, returning 0.0")
            return 0.0
    
    def print_results(self, score: float, method_name: str):
        """Print evaluation results in a formatted way."""
        print(f"\n=== Results for {method_name} ===")
        print(f"Acc: {score:.4f}")
    
    def plot_roc_curves(self, output_dir: str):
        """Plot ROC curves for all methods."""
        plt.figure(figsize=(10, 8))
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot ROC curve for each method
        for method_name, predictions in self.all_predictions.items():
            labels = self.all_labels[method_name]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(labels, predictions)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, linewidth=2, label=f'{method_name} (AUC = {roc_auc_score(labels, predictions):.3f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        
        # Customize plot
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Memorization Detection Methods', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Add AUC values in the plot
        plt.text(0.02, 0.98, f'Total Methods: {len(self.all_predictions)}', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plot_path = os.path.join(output_dir, "roc_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves plot saved to: {plot_path}")
    
    def plot_auc_comparison(self, results: Dict[str, float], output_dir: str):
        """Plot bar chart comparing AUC scores across methods."""
        plt.figure(figsize=(12, 8))
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Filter out metadata (keys starting with _)
        plot_results = {k: v for k, v in results.items() if not k.startswith('_')}
        
        # Prepare data for plotting
        methods = list(plot_results.keys())
        scores = list(plot_results.values())
        
        # Create bar chart
        bars = plt.bar(methods, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        plt.xlabel('Detection Methods', fontsize=12)
        plt.ylabel('AUC ROC Score', fontsize=12)
        
        # Add debug info to title if available
        debug_info = ""
        if "_debug" in results:
            debug_info = f" (DEBUG: {results['_dataset_size']} images, {results['_total_pairs']} pairs)"
        
        plt.title(f'AUC ROC Comparison Across Memorization Detection Methods{debug_info}', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if they're long
        plt.xticks(rotation=45, ha='right')
        
        # Add horizontal line at 0.5 (random classifier)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Classifier (AUC = 0.5)')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(output_dir, "auc_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"AUC comparison plot saved to: {plot_path}")
    
    def save_results(self, results: Dict[str, float], output_dir: str):
        """Save all results to a JSON file and create ROC curves plot."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numerical results
        output_path = os.path.join(output_dir, "benchmark_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Create and save ROC curves plot
        self.plot_roc_curves(output_dir)
        
        # Create and save AUC comparison plot
        self.plot_auc_comparison(results, output_dir)


    # Visualize example of robustness transform
def visualize_robustness_transform(dataset, output_dir, idx=0):
    """
    Visualize an example image pair showing original images and one with robustness transform.
    
    Args:
        dataset: MemorizationDetectionDataset instance
        output_dir: Directory to save visualization
        idx: Index of image pair to visualize
    """
    # Get paths for the image pair
    path1, path2, _ = dataset.all_pairs[idx]
    
    # Load both original images
    img1 = Image.open(os.path.join(dataset.base_dir, path1)).convert('RGB')
    img2 = Image.open(os.path.join(dataset.base_dir, path2)).convert('RGB')
    
    # Apply base transform to both images
    if dataset.transform:
        img1 = dataset.transform(img1)
        img2_orig = dataset.transform(img2)
        img2_robust = dataset.transform(img2)
    
    # Apply robustness transform to second image copy
    if dataset.robustness_transform:
        img2_robust = dataset.robustness_transform(img2_robust)
    
    # Convert tensors to numpy arrays and transpose to (H,W,C)
    img1_np = img1.permute(1,2,0).numpy()
    img2_orig_np = img2_orig.permute(1,2,0).numpy()
    img2_robust_np = img2_robust.permute(1,2,0).numpy()
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    # Plot images
    ax1.imshow(img1_np)
    ax1.set_title('Image 1 (Original)')
    ax1.axis('off')
    
    ax2.imshow(img2_orig_np)
    ax2.set_title('Image 2 (Original)')
    ax2.axis('off')
    
    ax3.imshow(img2_robust_np)
    ax3.set_title('Image 2 (With Robustness Transform)')
    ax3.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_transform_example.png'))
    plt.close()


def create_transforms(input_size: int = 512):
    """Create image transformations."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def save_side_by_side(img_left: torch.Tensor, img_right: torch.Tensor, out_path: str) -> None:
    """Save two CHW torch tensors side-by-side to disk as an image.

    Assumes inputs are in range [0, 1].
    """
    # Move to CPU, detach, convert to HWC numpy
    left = img_left.detach().cpu().permute(1, 2, 0).numpy()
    right = img_right.detach().cpu().permute(1, 2, 0).numpy()

    # Clip to valid range
    left = np.clip(left, 0.0, 1.0)
    right = np.clip(right, 0.0, 1.0)

    # Create figure
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(left)
    ax1.set_title('img1[0]')
    ax1.axis('off')
    ax2.imshow(right)
    ax2.set_title('img2[0]')
    ax2.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def load_thresholds(thresholds_path: str) -> Dict[str, float]:
    """
    Load detector thresholds from JSON file.
    
    Args:
        thresholds_path: Path to JSON file containing thresholds
        
    Returns:
        Dictionary mapping detector names to thresholds
    """
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")
    
    with open(thresholds_path, 'r') as f:
        thresholds = json.load(f)
    
    return thresholds

def apply_loaded_thresholds(detectors: List[BaseMemorizationDetector], thresholds: Dict[str, float]):
    """
    Apply loaded thresholds to detectors.
    
    Args:
        detectors: List of detector instances
        thresholds: Dictionary mapping detector names to thresholds
    """
    for detector in detectors:
        if detector.name in thresholds:
            detector._threshold = thresholds[detector.name]
            detector._is_fit = True
            print(f"Loaded threshold for {detector.name}: {detector._threshold}")
        else:
            print(f"Warning: No threshold found for detector {detector.name}")

def set_seed(seed): 
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def _to_display_tensor(img):
    """img: torch.Tensor [C,H,W] in [0,1] or [-1,1]. Returns [H,W,C] float32 in [0,1]."""
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            pass
        elif x.ndim == 2:
            x = x.unsqueeze(0)  # [1,H,W]
        else:
            raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
        # normalize if in [-1,1]
        if x.min() < 0:
            x = (x + 1) * 0.5
        x = x.clamp(0, 1)
        x = x.permute(1, 2, 0).numpy()  # [H,W,C]
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x
    else:
        raise TypeError("Pass a torch.Tensor")

def _apply_transform_safe(tfm, img_tensor):
    """Try tensor input first, else go through PIL."""
    try:
        out = tfm(img_tensor)
    except Exception:
        pil = transforms.ToPILImage()(img_tensor.cpu())
        out_pil = tfm(pil)
        out = transforms.ToTensor()(out_pil)
    return out

def save_transform_grid(transforms_dict, base_img_tensor, out_path="robustness_grid.png", dpi=120):
    """
    transforms_dict: {name: {'strengths': [...], 'make': lambda s: transform}}
    base_img_tensor: torch.Tensor [C,H,W], same image used for all transforms
    """
    torch.manual_seed(0)  # keep any stochastic ops fixed if ranges collapse to a single value
    cols = []
    max_rows = 0
    for tname, cfg in transforms_dict.items():
        strengths = list(cfg['strengths'])
        max_rows = max(max_rows, len(strengths))
        col_imgs = []
        for s in strengths:
            tfm = cfg['make'](s)
            out = _apply_transform_safe(tfm, base_img_tensor)
            col_imgs.append(_to_display_tensor(out))
        cols.append((tname, strengths, col_imgs))

    # Compute figure size (columns = transform types, rows = strength levels)
    h, w, _ = cols[0][2][0].shape
    n_cols = len(cols)
    n_rows = max_rows
    cell_w = 2.2
    cell_h = 1.6
    fig_w = max(6, n_cols * cell_w)
    fig_h = max(3.5, n_rows * cell_h)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Plot
    for c, (tname, strengths, imgs) in enumerate(cols):
        axes[0, c].set_title(tname)
        for r in range(n_rows):
            ax = axes[r, c]
            ax.axis("off")
            if r < len(imgs):
                ax.imshow(imgs[r])
                ax.text(0.5, -0.08, f"Strength {strengths[r]}",
                        transform=ax.transAxes, ha='center', va='top', fontsize=9)
            else:
                ax.set_facecolor((0, 0, 0))  # blank cell

    plt.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved grid to {out_path}")


def run_benchmark(csv_path: str, base_dir: str, output_dir: str, seed: int = 42, debug: bool = False, n_fit: int = None, n_test: int = None, sameimage: bool = False, load_thresholds_file: str = None, force_fit: bool = False, only_positive_test: bool = False, fit_method: str = 'percentile', fit_percentile: int = 5):
    """
    Run the complete benchmark evaluation.
    
    Args:
        csv_path: Path to CSV file with columns 'Split' and 'id'
        base_dir: Base directory for image paths
        output_dir: Directory to save results
        seed: Random seed for deterministic sampling
        debug: If True, limit dataset to 100 images for faster testing
        n_fit: Number of images from VAL split to fit on (before pairing)
        n_test: Number of images from TEST split to evaluate on (before pairing)
        sameimage: Whether to use same image for testing
        load_thresholds_file: Path to JSON file containing detector thresholds. 
                              If None, looks for detector_thresholds.json in output_dir.
        force_fit: If True, force fitting detectors even if thresholds file exists
        only_positive_test: If True and sameimage=False, test only on positive pairs (same identity). Default: True
        fit_method: Method for fitting thresholds ('percentile' or 'f1'). Default: 'percentile'
        fit_percentile: Percentile to use when fit_method='percentile' (default: 5)
    """
    print(f"Starting benchmark with:")
    print(f"CSV file: {csv_path}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Debug mode: {debug}")
    if n_fit is not None:
        print(f"Fit on first {n_fit} images of VAL split")
    if n_test is not None:
        print(f"Test on first {n_test} images of TEST split")
    
    if sameimage: 
        print("Using same image as input for each part of the Siamese network (Transforms are only applied to one).")
    elif only_positive_test:
        print("Test dataset will only include positive pairs (same identity).")
    
    # Create transforms
    transform = create_transforms()
    
    # Make run deterministic
    set_seed(0)

    os.makedirs(output_dir, exist_ok=True)
    
    # Check if thresholds file exists and can be loaded
    if load_thresholds_file is None:
        thresholds_path = os.path.join(output_dir, "detector_thresholds.json")
    else:
        thresholds_path = load_thresholds_file
    
    loaded_thresholds = None
    if not force_fit and os.path.exists(thresholds_path):
        try:
            loaded_thresholds = load_thresholds(thresholds_path)
            print(f"\nFound existing thresholds file: {thresholds_path}")
            print("Using loaded thresholds (skipping fitting). Use --force_fit to override.")
        except Exception as e:
            print(f"Warning: Could not load thresholds file: {e}")

    print("\nCreating validation dataset for fitting (includes positive and negative pairs)...")
    val_dataset = MemorizationDetectionDataset(csv_path, base_dir, "VAL", transform, seed=seed, debug=debug, max_images=n_fit, robustness_transform=transforms.Lambda(lambda x: x), sameimage=False)
 
    # Create dataloader
    # Use single-threaded loaders to avoid any nondeterministic worker behavior
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)
    
    # Visualize an example of robustness transform
    print("\nGenerating robustness transform visualization...")

    # Initialize detectors in the specified order
    if False: 
        mse = MeanSquaredErrorDetector()
        mse.name = "01_mse"
        detectors = [mse,]
    else:
        # ========== UNSUPERVISED ==========
        
        # 1. MSE
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

        
        # Assemble detectors in order
        detectors = [
            # Unsupervised
            mse,
            ssim,
            ncc,
            random_det,
            siam_ntxent_unsup,
            ## Foundation models
            byol,
            clip,
            convnext,
            data2vec,
            inception,
            mae,
            swav,
            dinov2,
            dinov3,
            ## Task-specific
            siam,
            siam_ntxent_task,
            ours,
            ours_ft_low,  # Commented out as it's replaced by siam
        ]

    # Fit detectors or load thresholds if available
    thresholds = {}
    
    # If thresholds file exists and user wants to skip fitting, load thresholds
    # Otherwise, fit detectors normally
    if loaded_thresholds is not None:
        print(f"\n{'='*50}")
        print("Loading thresholds from file (skipping fitting)...")
        apply_loaded_thresholds(detectors, loaded_thresholds)
        thresholds = loaded_thresholds
        print(f"{'='*50}")
    else:
        # Fit detectors and compute thresholds
        print(f"\nFitting detectors using method: {fit_method}")
        if fit_method == 'percentile':
            print(f"Using percentile: {fit_percentile}")
        for detector in detectors:
            print(f"\n{'='*50}")
            print(f"Fitting {detector.name}")
            detector.fit(val_loader, method=fit_method, percentile=fit_percentile)
            thresholds[detector.name] = float(detector._threshold) if detector._threshold is not None else None
            print(f"Threshold: {detector._threshold}")
            print(f"{'='*50}")
        
        # Save thresholds to JSON file
        thresholds_path = os.path.join(output_dir, "detector_thresholds.json")
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"\nThresholds saved to: {thresholds_path}")
        
    #print("Manually overwriting threshold for debugging purposes\n"*6)
    #ours_ft_low._threshold = 0.0062116873359973

    # Robustness transforms configuration: name -> strengths + builder
    def jpeg_compress(quality: int):
        from io import BytesIO
        def _fn(x):
            pil = transforms.ToPILImage()(x)
            buf = BytesIO()
            pil.save(buf, format='JPEG', quality=int(quality))
            buf.seek(0)
            out = Image.open(buf).convert('RGB')
            return transforms.ToTensor()(out)
        return _fn

    if False: 
        transforms_dict = {
            'Rotation': {
                'strengths': [0,30], # 5, 10, 15, 20, 30],
                'make': lambda deg: transforms.Lambda(lambda x: TF.rotate(x, angle=float(deg), expand=False)),
            },
        }
    else: 
        transforms_dict = {
            'GaussianBlur': {
                'strengths': [0.1, 1.0, 2.0, 6.0],
                'make': lambda s: transforms.GaussianBlur(kernel_size=9, sigma=float(s)),
            },
            'Noise': {
                'strengths': [0.01, 0.1, 0.2, 0.3],
                'make': lambda s: transforms.Lambda(lambda x: (x + torch.randn_like(x) * float(s)).clamp(0, 1)),
            },
            'Rotation': {
                'strengths': [5, 10, 20, 30],
                'make': lambda deg: transforms.Lambda(lambda x: TF.rotate(x, angle=float(deg), expand=False)),
            },
            'Brightness': {
                'strengths': [1.1, 1.3, 1.6, 2],
                'make': lambda f: transforms.ColorJitter(brightness=(float(f), float(f))),
            },
            'Contrast': {
                'strengths': [1.1, 1.3, 1.6, 2],
                'make': lambda f: transforms.ColorJitter(contrast=(float(f), float(f))),
            },
            'JPEGCompression': {
                # Define strength as compression strength (higher = stronger compression),
                # and map to JPEG quality = 100 - strength.
                'strengths': [5, 30, 60, 90],
                'make': lambda s: transforms.Lambda(jpeg_compress(max(1, 100 - int(s)))),
            },
        }

    viz_first = True

    # Load existing results if CSV exists
    results_csv_path = os.path.join(output_dir, "robustness_results.csv")
    if os.path.exists(results_csv_path):
        print(f"\nLoading existing results from: {results_csv_path}")
        try:
            df_existing = pd.read_csv(results_csv_path)
            # Check if CSV has required columns
            required_columns = ['Detector', 'Transform', 'Strength']
            if all(col in df_existing.columns for col in required_columns):
                print(f"Found {len(df_existing)} existing results")
                # Create a set of (Detector, Transform, Strength) tuples that have already been computed
                # Convert Strength to float for consistent comparison (handles int/float differences)
                existing_combinations = set(
                    (str(det), str(trans), float(strength)) 
                    for det, trans, strength in zip(
                        df_existing['Detector'], 
                        df_existing['Transform'], 
                        df_existing['Strength']
                    )
                )
                rows = df_existing.to_dict('records')
            else:
                print(f"Warning: CSV missing required columns {required_columns}. Starting fresh.")
                existing_combinations = set()
                rows = []
        except Exception as e:
            print(f"Warning: Could not load existing CSV: {e}")
            print("Starting fresh.")
            existing_combinations = set()
            rows = []
    else:
        print(f"\nNo existing results found. Starting fresh.")
        existing_combinations = set()
        rows = []

    for tname, cfg in transforms_dict.items():
        strengths = cfg['strengths']
        maker = cfg['make']
        print(f"\nEvaluating transform: {tname}")
        for strength in strengths:
            robustness_transform = maker(strength)
            # Create test dataset: use sameimage if specified, otherwise filter to only positive pairs
            test_dataset = MemorizationDetectionDataset(csv_path, base_dir, "TEST", transform, seed=seed, debug=debug, robustness_transform=robustness_transform, max_images=n_test, sameimage=sameimage)
            
            # If only_positive_test is True and sameimage is False, filter to only positive pairs
            if only_positive_test and not sameimage:
                # Filter to only positive pairs (label == 1)
                test_dataset.all_pairs = [(p1, p2, label) for p1, p2, label in test_dataset.all_pairs if label == 1]
                print(f"Filtered test dataset to {len(test_dataset.all_pairs)} positive pairs only")
            
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            if viz_first: 
                #visualizes the robustness the first time you go through
                base_img = test_dataset[0][0]  # tensor [3,H,W] in your code
                save_transform_grid(transforms_dict, base_img, out_path=os.path.join(output_dir,"robustness_grid.png"))
                viz_first = False

            # Cache transformed batches once to ensure identical inputs across detectors
            for detector in detectors:
                # Check if this combination has already been computed
                # Convert strength to float for consistent comparison
                combination = (str(detector.name), str(tname), float(strength))
                if combination in existing_combinations:
                    print(f"Skipping {detector.name} @ {tname}={strength} (already computed)")
                    continue
                
                print(f"Evaluating {detector.name} @ {tname}={strength}")

                # Set seed before each detector to ensure identical random transforms
                set_seed(0)
                with torch.no_grad():
                    preds, labels = detector.predict_batch(test_loader, show_progress=True)
                    preds = (torch.tensor(preds) > detector._threshold).int()
                acc = accuracy_score(labels, preds)
                prec = precision_score(labels, preds)
                rec = recall_score(labels, preds)
                f1 = f1_score(labels, preds)
                new_row = {
                    'Detector': detector.name,
                    'Transform': tname,
                    'Strength': strength,
                    'Recall': rec,
                    'Precision': prec,
                    'Accuracy': acc,
                    'F1': f1,
                }
                rows.append(new_row)
                existing_combinations.add(combination)

        # Save dataframe to disk after each transform (incremental saving)
        df = pd.DataFrame(rows)
        df.to_csv(results_csv_path, index=False)
        print(f"Saved results to: {results_csv_path} (total: {len(df)} rows)")

        # Plot per-transform recall across detectors
        df_transform = df[df['Transform'] == tname]
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_transform, x='Strength', y='Recall', hue='Detector', marker='o')
        plt.title(f'{tname}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"recall_vs_{tname}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
    


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark memorization detection methods using AUC ROC")
    parser.add_argument("--filelist", type=str, required=True, 
                       help="CSV file with columns 'Split' and 'id'")
    parser.add_argument("--basedir", type=str, required=True,
                       help="Base directory for image paths")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for deterministic sampling")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (limit to 100 images for faster testing)")
    parser.add_argument("--n_fit", type=int, default=None,
                       help="Number of images from VAL split to fit on (before pairing)")
    parser.add_argument("--n_test", type=int, default=None,
                       help="Number of images from TEST split to evaluate on (before pairing)")
    parser.add_argument("--sameimage", action="store_true", default=False,
                       help="Whether to use same image for testing")
    parser.add_argument("--only_positive_test", action="store_true", default=False,
                       help="If False and sameimage=False, test on both positive and negative pairs. Default: False (test only on positive pairs when sameimage=False)")
    parser.add_argument("--load_thresholds_file", type=str, default=None,
                       help="Path to JSON file containing detector thresholds. If not specified, looks for detector_thresholds.json in output_dir.")
    parser.add_argument("--force_fit", action="store_true", default=False,
                       help="Force fitting detectors even if thresholds file exists")
    parser.add_argument("--fit_method", type=str, default='percentile', choices=['percentile', 'f1'],
                       help="Method for fitting thresholds: 'percentile' (default) or 'f1' (maximizes F1 score)")
    parser.add_argument("--fit_percentile", type=int, default=5,
                       help="Percentile to use when fit_method='percentile' (default: 5)")
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.filelist):
        raise FileNotFoundError(f"CSV file not found: {args.filelist}")
    
    if not os.path.exists(args.basedir):
        raise FileNotFoundError(f"Base directory not found: {args.basedir}")
    
    # Run benchmark
    run_benchmark(args.filelist, args.basedir, args.output_dir, args.seed, args.debug, args.n_fit, args.n_test, args.sameimage, args.load_thresholds_file, args.force_fit, only_positive_test=args.only_positive_test, fit_method=args.fit_method, fit_percentile=args.fit_percentile)


if __name__ == "__main__":
    main()
