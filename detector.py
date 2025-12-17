import os
import sys
import pickle
import os
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
# Lazy import for torchmetrics - only SSIMDetector uses it
# from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple


class BaseMemorizationDetector:
    """Base class for all memorization detection methods."""
    
    def __init__(self, name: str):
        self.name = name
        self._is_fit = False 
        self._threshold = None
    
    def fit(self, dataloader, method='percentile', percentile=5):
        """
        Fit threshold based on validation dataset.
        
        Args:
            dataloader: DataLoader with image pairs and labels
            method: 'percentile' or 'f1'. If 'percentile', uses percentile-based threshold.
                    If 'f1', finds threshold that maximizes F1 score.
            percentile: Percentile to use when method='percentile' (default: 5, meaning 5th percentile of positive scores)
        """
        all_scores = []
        all_labels = []
        
        # Collect scores for all pairs
        for batch in tqdm(dataloader, desc="Fitting"):
            # Support datasets returning either 3-tuple (img1,img2,label) or 5-tuple (..., path1, path2)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    img1, img2, label = batch[0], batch[1], batch[2]
                else:
                    raise ValueError("Unexpected batch structure in fit()")
            else:
                raise ValueError("Unexpected batch type in fit()")
            with torch.no_grad():
                scores = self.predict(img1, img2)
                all_scores.extend(scores.tolist())
                all_labels.extend(label.tolist())
                
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if method == 'f1':
            # Find threshold that maximizes F1 score
            self._threshold = self._find_best_f1_threshold(all_scores, all_labels)
            print(f"Fitted threshold (F1-based): {self._threshold:.6f}")
        elif method == 'percentile':
            # Get scores for positive pairs (label=1)
            neg_scores = all_scores[all_labels == 0]
            # Set threshold at specified percentile of negative scores
            self._threshold = np.percentile(neg_scores, percentile)
            print(f"Fitted threshold (percentile-based, {percentile}th percentile): {self._threshold:.6f}")
        else:
            raise ValueError(f"Unknown fit method: {method}. Use 'percentile' or 'f1'.")
        
        self._is_fit = True
    
    def _find_best_f1_threshold(self, scores, labels):
        """
        Find the threshold that maximizes F1 score on validation data.
        
        Args:
            scores: Array of similarity scores
            labels: Array of true labels (0 or 1)
            
        Returns:
            Best threshold value
        """
        # Generate candidate thresholds from score range
        min_score = scores.min()
        max_score = scores.max()
        
        # Use a reasonable number of candidate thresholds
        # Try percentiles and evenly spaced values
        n_candidates = 100
        candidate_thresholds = np.linspace(min_score, max_score, n_candidates)
        
        # Also include percentile-based thresholds for better coverage
        percentiles = np.linspace(0, 100, 50)
        percentile_thresholds = np.percentile(scores, percentiles)
        candidate_thresholds = np.unique(np.concatenate([candidate_thresholds, percentile_thresholds]))
        
        best_f1 = -1
        best_threshold = None
        
        # Evaluate each candidate threshold
        for threshold in candidate_thresholds:
            predictions = (scores > threshold).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        if best_threshold is None:
            # Fallback to median if no threshold found
            best_threshold = np.median(scores)
            print(f"Warning: No threshold found, using median: {best_threshold:.6f}")
        else:
            print(f"Best F1 score: {best_f1:.4f} at threshold: {best_threshold:.6f}")
        
        return float(best_threshold)

    def predict_binary(self, img1: torch.Tensor, img2: torch.Tensor) -> int: 
        """ predict label 0 different 1 same"""
        if not self._is_fit: 
            return ValueError("Fit the model to training data to find the threshold first.")
        sim = self.predict(img1, img2)
        sim = int(sim > self._threshold)
        return sim

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Predict whether two images are from the same identity. Higher = more similar
        
        Args:
            img1: Image tensor [B,C,H,W] or [C,H,W]
            img2: Image tensor [B,C,H,W] or [C,H,W]
            
        Returns:
            Similarity scores tensor [B] (higher = more similar)
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def predict_batch_binary(self, dataloader: DataLoader, show_progress: bool = True): 
        predictions = []
        true_labels = []
        
        # Create progress bar if requested
        if show_progress:
            total_batches = len(dataloader)
            pbar = tqdm(total=total_batches, desc=f"Predicting binary with {self.name}", unit="batch")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        img1, img2, labels = batch[0], batch[1], batch[2]
                    else:
                        raise ValueError("Unexpected batch structure in predict_batch_binary()")
                else:
                    raise ValueError("Unexpected batch type in predict_batch_binary()")
                
                scores = self.predict(img1, img2)
                preds = (scores > self._threshold).int()
                predictions.extend(preds.tolist())
                true_labels.extend(labels.tolist())
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix({"pairs": len(predictions)})
        
        # Close progress bar
        if show_progress:
            pbar.close()
                    
        return predictions, true_labels

    def predict_batch(self, dataloader: DataLoader, show_progress: bool = True) -> Tuple[List[float], List[int]]:
        """
        Predict on a batch of data.
        
        Args:
            dataloader: DataLoader with image pairs
            show_progress: Whether to show progress bar with tqdm
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        predictions = []
        true_labels = []
        
        # Create progress bar if requested
        if show_progress:
            total_batches = len(dataloader)
            pbar = tqdm(total=total_batches, desc=f"Predicting with {self.name}", unit="batch")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        img1, img2, labels = batch[0], batch[1], batch[2]
                    else:
                        raise ValueError("Unexpected batch structure in predict_batch()")
                else:
                    raise ValueError("Unexpected batch type in predict_batch()")
                
                scores = self.predict(img1, img2)
                predictions.extend(scores.tolist())
                true_labels.extend(labels.tolist())
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix({"pairs": len(predictions)})
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        return predictions, true_labels


class DinoBackboneSiamese(BaseMemorizationDetector):
    def __init__(self,
                model_path: str,
                model_id: str,
                layers: int = 3,
                ):
            super().__init__("DinoBackboneSiamese") 
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"model_path not found: {model_path}")
            # Load HF DINOv3 backbone and processor
            self.dino_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._threshold = 0.5
            self._is_fit = True 

            if "SiameseNetwork" in sys.modules:
                #same architecture but different implementations
                del sys.modules["SiameseNetwork"]
                for p in  sys.path:  
                    if "networks" in p:
                         sys.path.remove(p)
            sys.path.append(os.path.join(os.path.dirname(__file__), "src/privacy/networks"))

            from SiameseNetwork import SiameseNetwork
            from transformers import AutoImageProcessor, AutoModel

            backbone_cfg = {
                'model_id': model_id,
                'dino_layers': layers,
                'out_method': 'cls_only',
                'dropout': 0.0,
                'finetune_backbone': False,
            }

            self.net = SiameseNetwork(
                network='dinov3_hf',
                backbone_cfg=backbone_cfg,
            ).to(self.dino_device).eval()

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.processor.size = {
                "height": 512,
                "width": 512
            }

            state = torch.load(model_path, map_location=self.dino_device)
            if isinstance(state, dict) and not isinstance(state, torch.nn.Module):
                # common convention: either flat state_dict or nested under "state_dict"
                if "state_dict" in state:
                    self.net.load_state_dict(state["state_dict"], strict=False)
                else:
                    self.net.load_state_dict(state, strict=False)

    def fit(self, dataloader, method='percentile', percentile=5): 
        # automaticlly trained with sigmoid output layer
        pass

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.net(img1, img2)         # expected: batch of logits
            probs = torch.sigmoid(out).flatten()
        return probs 

    def predict_batch(self, dataloader: DataLoader, show_progress: bool = True) -> Tuple[List[float], List[int]]:
        dataloader.dataset.processor = self.processor
        ret = super().predict_batch(dataloader, show_progress)
        dataloader.dataset.processor = None 
        return ret


    def predict_batch_binary(self, dataloader: DataLoader, show_progress: bool = True): 
        dataloader.dataset.processor = self.processor
        ret = super().predict_batch_binary(dataloader, show_progress)
        dataloader.dataset.processor = None
        return ret


class ImageSpaceSiameseDetector(BaseMemorizationDetector):
    """
    Siamese detector that computes latents on-the-fly from images.
    Uses a VAE to encode images to latents, then applies sample-wise normalization,
    and finally passes the normalized latents to the Siamese network.
    """
    def __init__(self,
                 packhaus_model_path: str,
                 vae_model_path: str = "stabilityai/stable-diffusion-2",
                 packhaus_in_channels: int = 4,
                 packhaus_n_features: int = 512, 
                 image_size: int = 512,
                 latent_size: int = 64,
                 network: str = 'ConvNeXt-Tiny',
                 final_mean: float = 0.0,
                 final_std: float = 0.5,
                 final_std: float = 0.5,
                 path_to_network_code=os.path.join(os.path.dirname(__file__), "src/privacy/networks"),
                 ):
        super().__init__("ImageSpaceSiamese")
        if not os.path.exists(packhaus_model_path):
            raise FileNotFoundError(f"packhaus_model_path not found: {packhaus_model_path}")
        
        # Load VAE model for encoding images to latents
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae").to("cuda").eval()
        self.latent_size = latent_size
        
        if "SiameseNetwork" in sys.modules:
            del sys.modules["SiameseNetwork"]
            for p in sys.path:  
                if "networks" in p:
                    sys.path.remove(p)

        sys.path.append(path_to_network_code)
        from SiameseNetwork import SiameseNetwork

        # Device
        self.device = torch.device("cuda")
        
        # Sample-wise normalization parameters
        self.final_mean = final_mean
        self.final_std = final_std

        # Build Siamese model (expects latents)
        self.net = SiameseNetwork(
            network=network,
            in_channels=packhaus_in_channels,
            n_features=packhaus_n_features
        ).to(self.device).eval()
        self.image_size = image_size  # Target size after resizing latents
        self.resizer = torch.nn.functional.interpolate

        # Load Siamese weights
        state = torch.load(packhaus_model_path, map_location=self.device)
        if isinstance(state, dict) and not isinstance(state, torch.nn.Module):
            if "state_dict" in state:
                self.net.load_state_dict(state["state_dict"], strict=False)
            else:
                self.net.load_state_dict(state, strict=False)
        else:
            self.net = state.to(self.device).eval()

        self.net = self.net.eval()

    def _encode_to_latent(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using VAE.
        
        Args:
            img: Image tensor [B, C, H, W] in [0, 1] range
            
        Returns:
            Latent tensor [B, 4, H_latent, W_latent] on GPU
        """
        with torch.no_grad():
            # VAE expects images in [0, 1], ensure they are
            img = img.clamp(0.0, 1.0).to(self.device)
            # Encode to latent (stays on GPU)
            latent_dist = self.vae.encode(img)
            latent = latent_dist.latent_dist.sample()  # [B, 4, H_latent, W_latent] on GPU
            return latent

    def _apply_sample_wise_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sample-wise normalization to latents.
        Normalizes each sample using its own channel-wise statistics.
        
        Args:
            x: Tensor [B,C,H,W] or [C,H,W]
        
        Returns:
            Normalized tensor with mean=final_mean and std=final_std per channel
        """
        # Handle both batched and single tensor cases
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute statistics for each sample and channel across spatial dimensions
        # x: [B, C, H, W]
        channel_mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        channel_std = x.std(dim=(2, 3), keepdim=True)    # [B, C, 1, 1]
        
        # Avoid division by zero
        channel_std = torch.clamp(channel_std, min=1e-8)
        
        # Normalize: (x - mean) / std, then scale to target mean/std
        x_normalized = (x - channel_mean) / channel_std
        x_normalized = x_normalized * self.final_std + self.final_mean
        
        if squeeze_output:
            x_normalized = x_normalized.squeeze(0)
        
        return x_normalized

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Predict similarity between two images.
        Images are encoded to latents on-the-fly, normalized, then passed to Siamese network.
        
        Args:
            img1: Image tensor [B,C,H,W] or [C,H,W] in [0, 1]
            img2: Image tensor [B,C,H,W] or [C,H,W] in [0, 1]
            
        Returns:
            Similarity probability in [0,1]
        """
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Ensure images are on the correct device and in [0, 1] range
        img1 = img1.clamp(0.0, 1.0).to(self.device)
        img2 = img2.clamp(0.0, 1.0).to(self.device)
        
        # Encode images to latents on-the-fly (stays on GPU)
        latent1 = self._encode_to_latent(img1)  # [B, 4, H_latent, W_latent] on GPU
        latent2 = self._encode_to_latent(img2)  # [B, 4, H_latent, W_latent] on GPU
        
        # Apply sample-wise normalization (on GPU)
        latent1 = self._apply_sample_wise_normalization(latent1)
        latent2 = self._apply_sample_wise_normalization(latent2)
        
        with torch.no_grad():
            out = self.net(latent1.to(torch.float32), latent2.to(torch.float32))  # expected: batch of logits
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.sigmoid(out).flatten()
            return probs.cpu()
        
    #def fit(self, dataloader, method='percentile', percentile=5):
    #    # Automatically trained with sigmoid output layer
    #    self._threshold = 0.5
    #    self._is_fit = True


class SiameseDetector(BaseMemorizationDetector):
    def __init__(self,
                 packhaus_model_path: str,
                 packhaus_in_channels: int = 3,
                 packhaus_n_features: int = 128, 
                 image_size: int = 512,
                 network: str= 'ResNet-50',
                 ):
        super().__init__("PackhausSiamese")
        if not os.path.exists(packhaus_model_path):
            raise FileNotFoundError(f"packhaus_model_path not found: {packhaus_model_path}")
        if "SiameseNetwork" in sys.modules:
            #same architecture but different implementations
            del sys.modules["SiameseNetwork"]
            for p in  sys.path:  
                if "networks" in p:
                    sys.path.remove(p)

        sys.path.append(os.path.join(os.path.dirname(__file__), "src/privacy/networks"))
        from SiameseNetwork import SiameseNetwork

        # Device
        self.device = torch.device("cuda")

        # Build model
        self.net = SiameseNetwork(
            network=network,
            in_channels=packhaus_in_channels,
            n_features=packhaus_n_features
        ).to(self.device).eval()
        self.image_size = image_size 
        self.resizer = torch.nn.functional.interpolate

        # Load weights (accepts raw state_dict or a saved nn.Module)
        state = torch.load(packhaus_model_path, map_location=self.device)
        if isinstance(state, dict) and not isinstance(state, torch.nn.Module):
            # common convention: either flat state_dict or nested under "state_dict"
            if "state_dict" in state:
                self.net.load_state_dict(state["state_dict"], strict=False)
            else:
                self.net.load_state_dict(state, strict=False)
        else:
            # If someone saved the whole module
            self.net = state.to(self.device).eval()

        if packhaus_in_channels == 3: 
            self.transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else: 
            self.transform_norm = lambda x: x
        self.net = self.net.eval()

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        img1, img2 come from your DataLoader transforms: [B,C,H,W] or [C,H,W] in float tensors.
        Returns similarity probability in [0,1].
        """
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        if img1.size()[-1] != self.image_size: 
            imgs = self.resizer(torch.cat([img1, img2]), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            img1, img2 = imgs.chunk(2)

        img1, img2 = self.transform_norm(torch.cat([img1, img2])).chunk(2)

        with torch.no_grad():
            x1 = img1.to(self.device, dtype=torch.float32)
            x2 = img2.to(self.device, dtype=torch.float32)
            out = self.net(x1, x2)         # expected: batch of logits
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.sigmoid(out).flatten()
            return probs.cpu()
        
    #def fit(self, dataloader, method='percentile', percentile=5): 
    #    # automaticlly trained with sigmoid output layer
    #    self._threshold = 0.5
    #    self._is_fit = True


class NCCDetector(BaseMemorizationDetector):
    """
    Normalized cross-correlation over flattened pixels.
    Returns value in [-1, 1], higher = more similar.
    """
    def __init__(self):
        super().__init__("NCC")

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            
        # Flatten spatial dims: [B,C,H,W] -> [B,C*H*W]
        x = img1.flatten(start_dim=1).to(dtype=torch.float32)
        y = img2.flatten(start_dim=1).to(dtype=torch.float32)

        # Center
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)

        # Compute correlation
        num = (x * y).sum(dim=1)
        den = torch.sqrt((x * x).sum(dim=1) * (y * y).sum(dim=1)) + 1e-12
        return num / den


class DarSiameseDetector(SiameseDetector):
    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        img1, img2 come from your DataLoader transforms: [B,C,H,W] or [C,H,W] in float tensors.
        Returns similarity probability in [0,1].
        """
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        if img1.size()[-1] != self.image_size: 
            imgs = self.resizer(torch.cat([img1, img2]), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            img1, img2 = imgs.chunk(2)

        img1, img2 = self.transform_norm(torch.cat([img1, img2])).chunk(2)

        with torch.no_grad():
            x1 = img1.to(self.device, dtype=torch.float32)
            x2 = img2.to(self.device, dtype=torch.float32)
            out_x1 = self.net.forward_once(x1)
            out_x2 = self.net.forward_once(x2)

        sim = torch.from_numpy(-cdist(out_x1.cpu().numpy(), out_x2.cpu().numpy(), metric="correlation"))
        return sim.diagonal()

    def fit(self, dataloader, method='percentile', percentile=5):
        """
        Fit threshold based on validation dataset.
        
        Args:
            dataloader: DataLoader with image pairs and labels
            method: 'percentile' or 'f1'. If 'percentile', uses percentile-based threshold.
                    If 'f1', finds threshold that maximizes F1 score.
            percentile: Percentile to use when method='percentile' (default: 5, meaning 5th percentile of positive scores)
        """
        all_scores = []
        all_labels = []
        
        # Collect scores for all pairs
        for batch in tqdm(dataloader, desc="Fitting"):
            # Support datasets returning either 3-tuple (img1,img2,label) or 5-tuple (..., path1, path2)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    img1, img2, label = batch[0], batch[1], batch[2]
                else:
                    raise ValueError("Unexpected batch structure in fit()")
            else:
                raise ValueError("Unexpected batch type in fit()")
            with torch.no_grad():
                scores = self.predict(img1, img2)
                all_scores.extend(scores.tolist())
                all_labels.extend(label.tolist())
                
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        if method == 'f1':
            # Find threshold that maximizes F1 score
            self._threshold = self._find_best_f1_threshold(all_scores, all_labels)
            print(f"Fitted threshold (F1-based): {self._threshold:.6f}")
        elif method == 'percentile':
            # Get scores for positive pairs (label=1)
            pos_scores = all_scores[all_labels == 1]
            # Set threshold at specified percentile of positive scores
            self._threshold = np.percentile(pos_scores, percentile)
            print(f"Fitted threshold (percentile-based, {percentile}th percentile): {self._threshold:.6f}")
        else:
            raise ValueError(f"Unknown fit method: {method}. Use 'percentile' or 'f1'.")
        
        self._is_fit = True


class MemLDM2DDetector(BaseMemorizationDetector):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__("MemLDM2D")
        self.device = torch.device("cuda")
        self.resizer = torch.nn.functional.interpolate
        if "SiameseNetwork" in sys.modules:
            #same architecture but different implementations
            del sys.modules["SiameseNetwork"]
            for p in  sys.path:  
                if "networks" in p:
                    sys.path.remove(p)

        sys.path.append(os.path.join(os.path.dirname(__file__), "src/privacy/networks"))
        from SiameseNetwork import SiameseNetwork

        self.image_size = 512
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Predict similarity between two images."""
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        if img1.size()[-1] != self.image_size:
            img1 = self.resizer(img1, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            img2 = self.resizer(img2, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # Mean across channels and compute embeddings in batch
            emb1 = self.model(img1.mean(dim=1, keepdim=True).to(self.device), resnet_only=True)
            emb2 = self.model(img2.mean(dim=1, keepdim=True).to(self.device), resnet_only=True)
            
            # Compute correlation distance between embeddings
            sim = torch.from_numpy(-cdist(emb1.cpu().numpy(), emb2.cpu().numpy(), metric="correlation"))
            return sim.diagonal()


class MeanSquaredErrorDetector(BaseMemorizationDetector):
    """Memorization detector using Mean Squared Error between images."""
    
    def __init__(self):
        super().__init__("MeanSquaredError")
    
    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between two images. Lower MSE means more similar (same identity).
        We return negative MSE so that higher values indicate more similarity.
        """
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            
        mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
        return -mse  # Negative so higher = more similar


class BeyondFIDDetector(BaseMemorizationDetector):
    def __init__(self, model_name: str="clip"):
        super().__init__(model_name)
        from beyondfid.feature_extractor_models import load_feature_model, _FEATURE_MODELS
        from beyondfid.default_config import config
        config = config.feature_extractors.get(model_name)
        if config is None:
            raise ValueError(f"Model: {model_name} not found. Available models: {list(_FEATURE_MODELS.keys())}")

        if model_name == "byol": 
            config.config.model_path = "/vol/ideadata/ed52egek/pycharm/improve_recall/BeyondFID/beyondfid/feature_extractor_models/byol/large_model.pth"
            config.config.cfg_path = "/vol/ideadata/ed52egek/pycharm/improve_recall/BeyondFID/beyondfid/feature_extractor_models/byol/config_large.yaml"

        self.model = load_feature_model(config).to("cuda")

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if img1.dim() == 3:
            images = torch.stack([img1, img2])
        else: 
            images =  torch.cat([img1, img2])

        # uses for loop 
        feats = []
        feats = self.model(images.to("cuda"))
        img1_feat, img2_feat = feats.chunk(2)

        return np.diag(cosine_similarity(img1_feat.cpu().numpy(), img2_feat.cpu().numpy()))



class DenseClassifierDetector(BaseMemorizationDetector):
    """
    Memorization detector that reuses DenseNet classifier checkpoints generated by
    `compute_classifier_features.py`. It extracts penultimate-layer features from a
    single DenseNet model (no ensemble) and compares them with cosine similarity.

    The checkpoint path is automatically resolved based on the directory structure
    used in `compute_all_clf_features.sh`.
    """

    DATASET_ALIASES = {
        "celeba": "celeba",
        "cxr-lt": "cxr-lt",
        "mimic": "mimic",
        "isic": "isic_clean",
        "isic_clean": "isic_clean",
        "imagenet_lt": "imagenet_lt",
        "ctrate": "ctrate_slices_final",
        "ctrate_slices_final": "ctrate_slices_final",
    }

    class DenseNetFeatureExtractor(nn.Module):
        """Wrapper around DenseNet121 that extracts features from the classifier head or penultimate layer."""

        def __init__(self, num_classes: int, use_outlogits: bool = True):
            super().__init__()
            self.model = models.densenet121(weights=None, num_classes=num_classes)
            self.use_outlogits = use_outlogits
            self.feature_dim = self.model.classifier.in_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.use_outlogits:
                features = self.model.features(x)
                out = torch.nn.functional.relu(features, inplace=True)
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                out = torch.flatten(out, 1)
            else:
                out = torch.sigmoid(self.model(x))
            return out

        def load_checkpoint(self, checkpoint_path: str, device: str = "cuda"):
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.eval()
            return self

    def __init__(
        self,
        dataset_name: str,
        *,
        model_path: Optional[str] = None,
        model_base_dir: Optional[str] = None,
        filelist_csv: Optional[str] = None,
        split: Optional[str] = None,
        image_size: int = 512,
        use_outlogits: bool = False,
        device: Optional[str] = None,
        include_ground_truth: bool = False,
    ):
        super().__init__("DenseClassifier")

        self.dataset_name = dataset_name.replace(".csv", "")
        self.image_size = image_size
        self.use_outlogits = use_outlogits
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split = split
        self.include_ground_truth = include_ground_truth
        self.filelist_csv = filelist_csv

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        workspace_root = Path(__file__).resolve().parents[1]
        default_base = workspace_root / "outputs" / "full_downstream_isam" / "full_models"
        self.model_base_dir = Path(model_base_dir) if model_base_dir else default_base
        self.model_path = Path(model_path) if model_path else self._resolve_model_path()

        state_dict = self._load_state_dict(self.model_path)
        num_classes = self._infer_num_classes(state_dict)

        self.model = self.DenseNetFeatureExtractor(num_classes, use_outlogits=self.use_outlogits)
        self.model.load_checkpoint(str(self.model_path), device=str(self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_dim = self.model.feature_dim
        self.label_map: Optional[Dict[str, torch.Tensor]] = None
        self.label_columns: Optional[List[str]] = None

        if self.include_ground_truth:
            if not self.filelist_csv:
                raise ValueError("filelist_csv must be provided when include_ground_truth=True.")
            self._prepare_label_map()

    def _resolve_model_path(self) -> Path:
        key = self.dataset_name.lower()
        alias = self.DATASET_ALIASES.get(key)
        if alias is None:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported for DenseClassifierDetector.")
        model_dir = self.model_base_dir / f"{alias}_full"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Expected model directory '{model_dir}' for dataset '{self.dataset_name}' does not exist.")

        candidates = sorted(model_dir.glob("densenet121_best_*.pth"))
        if not candidates:
            candidates = sorted(model_dir.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No DenseNet checkpoints found in '{model_dir}'.")
        return candidates[-1]

    @staticmethod
    def _load_state_dict(model_path: Path) -> dict:
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            return state["state_dict"]
        if isinstance(state, dict):
            return state
        return state.state_dict()

    @staticmethod
    def _infer_num_classes(state_dict: dict) -> int:
        for key, value in state_dict.items():
            if key.endswith("classifier.weight"):
                return value.shape[0]
        raise ValueError("Unable to infer number of classes from checkpoint (classifier.weight not found).")

    def _prepare_label_map(self) -> None:
        df = pd.read_csv(self.filelist_csv)  # type: ignore[arg-type]
        df = df.drop(columns={"Unnamed: 0", "Unnamed: 0.1", "fold"}, errors="ignore")
        if self.split:
            df = df[df["Split"] == self.split].reset_index(drop=True)
        label_columns = [c for c in df.columns if c not in {"id", "Split", "path"}]
        if not label_columns:
            raise ValueError("No label columns found in filelist CSV for DenseClassifierDetector.")
        self.label_columns = label_columns
        self.label_map = {}
        for _, row in df.iterrows():
            path = row["path"]
            labels = torch.tensor(row[label_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
            self.label_map[path] = labels

    def _lookup_labels(self, paths: Optional[List[str]]) -> Optional[torch.Tensor]:
        if not self.include_ground_truth or paths is None or self.label_map is None:
            return None

        tensors = []
        num_labels = len(self.label_columns) if self.label_columns else 0
        for path in paths:
            labels = self.label_map.get(path)
            if labels is None:
                labels = torch.zeros(num_labels, dtype=torch.float32)
            tensors.append(labels)
        return torch.stack(tensors, dim=0).to(self.device)

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device, dtype=torch.float32)
        if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
            img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img

    @torch.inference_mode()
    def _extract_features(self, img: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        img = self._preprocess(img)
        features = self.model(img)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        if labels is not None:
            if labels.shape[1] != features.shape[1]:
                raise ValueError(
                    f"Label dimension {labels.shape[1]} does not match feature dimension {features.shape[1]} for DenseClassifierDetector."
                )
            features = features + labels
        return F.normalize(features, dim=1)

    @torch.inference_mode()
    def _predict_with_paths(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        paths1: Optional[List[str]] = None,
        paths2: Optional[List[str]] = None,
    ) -> torch.Tensor:
        labels1 = self._lookup_labels(paths1)
        labels2 = self._lookup_labels(paths2)
        feat1 = self._extract_features(img1, labels1)
        feat2 = self._extract_features(img2, labels2)
        sims = F.cosine_similarity(feat1, feat2, dim=1)
        return sims.cpu()

    @torch.inference_mode()
    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self._predict_with_paths(img1, img2, None, None)

    @torch.inference_mode()
    def predict_batch(self, dataloader: DataLoader, show_progress: bool = True) -> Tuple[List[float], List[int]]:
        predictions: List[float] = []
        true_labels: List[int] = []

        dataset = dataloader.dataset
        pairs = getattr(dataset, "pairs", None)
        idx_cursor = 0

        pbar = tqdm(total=len(dataloader), desc=f"Predicting with {self.name}", unit="batch") if show_progress else None

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                img1, img2, labels = batch[0], batch[1], batch[2]
            else:
                raise ValueError("Unexpected batch structure in DenseClassifierDetector.predict_batch()")

            batch_size = img1.size(0)
            batch_paths1 = batch_paths2 = None
            if pairs is not None and len(pairs) >= idx_cursor + batch_size:
                batch_pairs = pairs[idx_cursor: idx_cursor + batch_size]
                batch_paths1 = [p[0] for p in batch_pairs]
                batch_paths2 = [p[1] for p in batch_pairs]
            elif self.include_ground_truth:
                raise ValueError(
                    "Ground truth augmentation requires deterministic pairs with path information. "
                    "Run the benchmark in TEST mode where dataset.pairs is defined."
                )

            scores = self._predict_with_paths(img1, img2, batch_paths1, batch_paths2)
            predictions.extend(scores.tolist())
            true_labels.extend(labels.cpu().tolist())

            idx_cursor += batch_size
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"pairs": len(predictions)})

        if pbar is not None:
            pbar.close()

        return predictions, true_labels


class DenseClassifierLabelsOnlyDetector(BaseMemorizationDetector):
    """
    Labels-only variant: ignores pixel inputs and computes similarity purely from
    ground-truth label vectors loaded from the dataset CSV. This mirrors the
    'noisygt' and GT-augmented pipeline style by using only GT information.
    """
    DATASET_ALIASES = DenseClassifierDetector.DATASET_ALIASES

    def __init__(
        self,
        dataset_name: str,
        *,
        filelist_csv: str,
        split: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__("DenseClassifierLabelsOnly")
        self.dataset_name = dataset_name.replace(".csv", "")
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split = split
        self.label_map: Dict[str, torch.Tensor] = {}
        self.label_columns: List[str] = []
        self._build_label_map(filelist_csv)

    def _build_label_map(self, filelist_csv: str) -> None:
        df = pd.read_csv(filelist_csv)
        df = df.drop(columns={"Unnamed: 0", "Unnamed: 0.1", "fold"}, errors="ignore")
        if self.split:
            df = df[df["Split"] == self.split].reset_index(drop=True)
        label_columns = [c for c in df.columns if c not in {"id", "Split", "path"}]
        if not label_columns:
            raise ValueError("No label columns found in filelist CSV for DenseClassifierLabelsOnlyDetector.")
        self.label_columns = label_columns
        for _, row in df.iterrows():
            path = row["path"]
            labels = torch.tensor(row[label_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
            self.label_map[path] = labels.to(self.device)

    @torch.inference_mode()
    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Labels-only detector requires path context; single-call predict can't map labels reliably.
        raise RuntimeError("DenseClassifierLabelsOnlyDetector.predict requires path information; use predict_batch with a dataset that exposes pairs.")

    @torch.inference_mode()
    def predict_batch(self, dataloader: DataLoader, show_progress: bool = True) -> Tuple[List[float], List[int]]:
        predictions: List[float] = []
        true_labels: List[int] = []

        dataset = dataloader.dataset
        pairs = getattr(dataset, "pairs", None)
        if pairs is None:
            raise ValueError("DenseClassifierLabelsOnlyDetector requires dataset.pairs to be available (TEST mode).")

        idx_cursor = 0
        pbar = tqdm(total=len(dataloader), desc=f"Predicting with {self.name}", unit="batch") if show_progress else None

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                _, _, labels = batch[0], batch[1], batch[2]
            else:
                raise ValueError("Unexpected batch structure in DenseClassifierLabelsOnlyDetector.predict_batch()")

            batch_size = labels.size(0)
            batch_pairs = pairs[idx_cursor: idx_cursor + batch_size]
            paths1 = [p[0] for p in batch_pairs]
            paths2 = [p[1] for p in batch_pairs]

            lbl1 = torch.stack([self.label_map.get(p, torch.zeros(len(self.label_columns), device=self.device)) for p in paths1], dim=0)
            lbl2 = torch.stack([self.label_map.get(p, torch.zeros(len(self.label_columns), device=self.device)) for p in paths2], dim=0)

            # Cosine similarity between label vectors
            lbl1 = torch.nn.functional.normalize(lbl1, dim=1)
            lbl2 = torch.nn.functional.normalize(lbl2, dim=1)
            sims = torch.sum(lbl1 * lbl2, dim=1)  # cosine since normalized

            predictions.extend(sims.detach().cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

            idx_cursor += batch_size
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"pairs": len(predictions)})

        if pbar is not None:
            pbar.close()

        return predictions, true_labels


class SSIMDetector(BaseMemorizationDetector):
    def __init__(self, data_range: float = 1.0):
        super().__init__("SSIM")
        self.data_range = data_range

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Lazy import to avoid dependency issues when only using other detectors
        from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
        
        # Add batch dim if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            
        # Expect [B,C,H,W] in [0, data_range]
        scores = tm_ssim(
            img1,
            img2,
            data_range=self.data_range,
            reduction=None
        )
        if len(scores) == 1: 
            scores = scores[0].item()
        else: 
            scores = torch.tensor(scores)
        return scores


class Dinov3Detector(BaseMemorizationDetector):
    """Memorization detector using Dinov3 convnext."""
    def __init__(self, name, layer, image_size=512, ckpt="facebook/dinov3-vits16-pretrain-lvd1689m"):
        super().__init__(name)
        from transformers import AutoImageProcessor, AutoModel
        # pick any convnext checkpoint, e.g. tiny
        self.proc = AutoImageProcessor.from_pretrained(ckpt)
        self.proc.size = {"height": image_size, "width": image_size}
        self.model = AutoModel.from_pretrained(ckpt).to("cuda").eval()  # returns DINOv3ConvNextModel
        self.layer = layer
    
    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        """
        # Add batch dim if needed
        if img1.ndim == 3: 
            img1 = img1.unsqueeze(dim=0)
            img2 = img2.unsqueeze(dim=0)

        with torch.inference_mode():
            out = self.model(torch.cat([img1.to("cuda"), img2.to("cuda")]), output_hidden_states=True)
            if self.layer == 0: 
                outputs = out["last_hidden_state"].view(len(img1) * 2, -1)  # [2B, D]
            elif self.layer == 1: 
                outputs = out["pooler_output"].view(len(img1) * 2, -1)  # [2B, D]

            out1, out2 = outputs.cpu().chunk(2)  # Each [B, D]
            
            # Compute cosine similarity for each pair
            sims = []
            for e1, e2 in zip(out1, out2):
                sim = 1 - float(cdist(e1.unsqueeze(0).detach().numpy(), 
                                    e2.unsqueeze(0).detach().numpy(), 
                                    metric="cosine")[0,0])
                sims.append(sim)
            return torch.tensor(sims)

    
    def fit(self, dataloader, method='percentile', percentile=5):
        # automaticlly trained with sigmoid output layer
        dataloader.dataset.processor = self.proc
        ret = super().fit(dataloader, method=method, percentile=percentile)
        dataloader.dataset.processor = None 
        return ret


    def predict_batch(self, dataloader: DataLoader, show_progress: bool = True) -> Tuple[List[float], List[int]]:
        dataloader.dataset.processor = self.proc
        ret = super().predict_batch(dataloader, show_progress)
        dataloader.dataset.processor = None 
        return ret


    def predict_batch_binary(self, dataloader: DataLoader, show_progress: bool = True): 
        dataloader.dataset.processor = self.proc
        ret = super().predict_batch_binary(dataloader, show_progress)
        dataloader.dataset.processor = None
        return ret


class Dinov3PrecomputedDetector(BaseMemorizationDetector):
    """Dino """
    def __init__(self, basedir, layer):
        super().__init__("Dinov3")
        self.basedir = basedir
        self.layer = layer
   
    def predict(self, path1, path2):

        # Load the pickle files
        with open(os.path.join(self.basedir, path1.replace(".png", ".pkl")), 'rb') as f:
            d1 = pickle.load(f)
        with open(os.path.join(self.basedir, path2.replace(".png", ".pkl")), 'rb') as f:
            d2 = pickle.load(f)
            
        d1 = d1[self.layer]
        d2 = d2[self.layer]
        
        # Reshape to 2D arrays for cdist and ensure single distance value
        d1_2d = d1.reshape(1, -1)  # [1, N]
        d2_2d = d2.reshape(1, -1)  # [1, N]
        return 1 - float(cdist(d1_2d, d2_2d, metric="cosine")[0,0])

    def predict_batch(self, dataloader: DataLoader, show_progress: bool = True) -> Tuple[List[float], List[int]]:
        """
        Predict on a batch of data.
        
        Args:
            dataloader: DataLoader with image pairs
            show_progress: Whether to show progress bar with tqdm
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        predictions = []
        true_labels = []
        
        # Create progress bar if requested
        if show_progress:
            total_batches = len(dataloader)
            pbar = tqdm(total=total_batches, desc=f"Predicting with {self.name}", unit="batch")
        
        with torch.no_grad():
            for batch_idx, (img1, _, labels, path1, path2) in enumerate(dataloader):
                batch_size = img1.size(0)
                for i in range(batch_size):
                    pred = self.predict(path1[i], path2[i])
                    predictions.append(pred)
                    true_labels.append(labels[i].item())
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix({"pairs": len(predictions)})
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        return predictions, true_labels
