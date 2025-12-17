from yaml import TagToken
from torch.utils import data
import torchio as tio
import pandas as pd
import numpy as np
import pickle
import random
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn 
import torch 
from tqdm import tqdm
import torch
import hashlib


class SimplePathDataset(data.Dataset):
    def __init__(self, paths, basedir, transform=None, n_channels=3, is_latent=False):
        self.paths = paths
        self.basedir = basedir
        self.transform = transform
        self.n_channels = n_channels
        self.is_latent = is_latent

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.is_latent:
            # Load latent tensor from disk
            file_path = os.path.join(self.basedir, path + ".pt")
            data = torch.load(file_path)
            return path, data
        else:
            # Load image and apply transform
            img = img_loader(self.basedir, path, self.n_channels)
            if self.transform:
                img = self.transform(img)
            return path, img


class SiameseNCEDataset(data.Dataset): 
    def __init__(self, phase='TRAIN', n_channels=3, transform=None, basedir='./', filelist='list.csv', aug_transform=None, processor=None, seed=42, unsupervised=False, is_latent=False, fitmode=False, distill=False):
        self.phase = phase
        self.df = pd.read_csv(filelist)
        self.basedir = basedir
        self.n_channels = n_channels
        self.transform = transform
        self.aug_transform = aug_transform # applied only to one image 
        self.processor = processor
        self.unsupervised = unsupervised
        self.is_latent = is_latent
        self.fitmode = fitmode
        self.distill = distill
        
        # Filter data based on phase
        self.df = self.df[self.df['Split'] == self.phase].reset_index(drop=True)
        self.ids = np.array(list(self.df["id"])) if hasattr(self.df, "id") and not self.unsupervised else np.arange(len(self.df))

        assert len(self.df["id"].unique()) >= 3, "we need at least three different identities to produce valid training input"
        
        # Initialize cache for latent mode (will be populated lazily)
        if self.is_latent:
            self.cached_data = {}
        
        # Create image pairs for siamese training
        # Group paths by ID and store as dictionary where each ID maps to list of paths
        self.paths = self.df.groupby('id')['path'].apply(list).reset_index()["path"].to_dict()

        # For test phase, create deterministic positive and negative pairs
        if phase == 'TEST' or self.fitmode:
            # Use generator to avoid interfering with other random seeds
            self.rng = random.Random(seed)
            
            # Store test data for pair generation
            self.pairs = []
            self.base_dataset = []
            
            if hasattr(self.df, "ds_name"): 
                self.sub_datasets = {ds: self.df[self.df.ds_name == ds] for ds in self.df.ds_name.unique().tolist()}
            else: 
                self.sub_datasets = {"test": self.df}

            # For each ID
            for ds_name, dsdf in self.sub_datasets.items(): 
                # sort by id to make finding positives easier
                test_paths_byid = dsdf.groupby('id')['path'].apply(list).reset_index()["path"].to_dict()

                for test_idx, test_paths_same_id in (test_paths_byid).items():

                    # For each image in this ID's paths
                    for i, path_x in enumerate(test_paths_same_id):
                        # Get one positive pair (same ID, different image if available)
                        if len(test_paths_same_id) > 1:
                            pos_path = test_paths_same_id[(i + 1) % len(test_paths_same_id)]

                            # positive pair added
                            self.pairs.append((path_x, pos_path, 1))
                            self.base_dataset.append(ds_name)
                            
                            # Get one negative pair (different ID)
                            # Create list of other IDs excluding current ID
                            neg_id_found = False
                            while not neg_id_found: 
                                neg_id = self.rng.randint(0, len(test_paths_byid)-1)
                                if neg_id == test_idx: 
                                    continue
                                neg_id_found = True
                                neg_paths = test_paths_byid[neg_id]
                                neg_path = neg_paths[0]  # Take first image from negative ID

                            # negative pair added
                            self.pairs.append((path_x, neg_path, 0))
                            self.base_dataset.append(ds_name)

        # Total number of samples
        self.n_samples = len(self.pairs) if (phase == 'TEST' or self.fitmode) else len(self.paths)
        if self.distill: 
            self._distill_cache = []

    def __len__(self):
        return self.n_samples

    def precopute_for_distillation(self, transform, model):
        """
        Precompute features from the model for all paths in the dataset (for distillation),
        storing them in self._distill_cache as a dict: path -> feature tensor.
        """

        if self.phase == 'TEST' or self.fitmode:
            raise NotImplementedError("Distillation for test phase not implemented")

        # Collect all paths from self.paths (can be list of lists or list of strings)
        # self.paths is a dict of lists of strings (e.g., {id1: [path1, path2], ...})
        all_paths = []
        for paths_list in self.paths.values():
            all_paths.extend(paths_list)

        # Remove duplicates and sort for consistency
        all_paths = sorted(list(set(all_paths)))

        # Build DataLoader for all paths
        batch_size = 32  # Reasonable batch size
        dataset = SimplePathDataset(all_paths, "/vol/ideadata/ed52egek/pycharm/syneverything/outputs/latents", transform, self.n_channels, True)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        model = model.to("cuda")

        # Hash for all_paths as a cache filename (use hex/str for usability)
        hash_key = hashlib.md5("".join(all_paths).encode("utf-8")).hexdigest()
        hash_path = os.path.join(self.basedir, f"distill_cache_{hash_key}.pt")

        if os.path.exists(hash_path):
            print(f"Loading precomputed logits from {hash_path}")
            _distill_cache = torch.load(hash_path)
        else:
            # Run inference and aggregate results
            _distill_cache = {}  # Temporary cache, key: path, value: feature tensor (on cpu)
            model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Precompute Dataset Distilled Features"):
                    path_list, inputs = batch
                    inputs = inputs.cuda()
                    feat = model.forward_once_logits(inputs)
                    feat_cpu = feat.detach().cpu()
                    for p, f in zip(path_list, feat_cpu):
                        _distill_cache[p] = f
            torch.save(_distill_cache, hash_path)
            print(f"Saved precomputed logits to {hash_path}")

        # Save to member cache
        self._distill_cache = _distill_cache
        model.train()

    def _load_and_cache(self, path):
        """Load and cache a single image/latent representation."""
        if path in self.cached_data:
            return self.cached_data[path]
        
        file_path = os.path.join(self.basedir, path + ".pt")
        try:
            data = torch.load(file_path)
            self.cached_data[path] = data
            return data
        except FileNotFoundError:
            print(f"Warning: Latent file not found: {file_path}")
            # Fallback to loading as image and converting
            img = pil_loader(self.basedir, path, self.n_channels)
            if self.transform:
                img = self.transform(img)
            self.cached_data[path] = img
            return img

    def __getitem__(self, index):
        if self.is_latent: 
            file_ending = ".pt"
        else: 
            file_ending = ""

        if self.phase == 'TEST' or self.fitmode:
            # For test phase, use pre-generated pairs
            path_x, path_xd, label = self.pairs[index]
            
            if self.is_latent:
                x1 = self._load_and_cache(path_x)
                x2 = self._load_and_cache(path_xd)
            else:
                x1 = img_loader(self.basedir, path_x + file_ending, self.n_channels)
                x2 = img_loader(self.basedir, path_xd + file_ending, self.n_channels)
        else:
            # Training/validation phase - random sampling
            tgt_idx = index.item()
            paths = self.paths[tgt_idx] 
            if isinstance(paths, str): 
                paths = [paths,]

            path_x = paths[torch.randint(high=len(paths), size=(1,)).item()]
            path_xd = paths[torch.randint(high=len(paths), size=(1,)).item()]

            if self.distill: 
                x1 = img_loader(self.basedir, path_x + file_ending, self.n_channels)
                x2 = self._distill_cache[path_xd]
            elif self.is_latent:
                x1 = self._load_and_cache(path_x)
                if path_x == path_xd: 
                    x2 = x1.clone()
                else: 
                    x2 = self._load_and_cache(path_xd)
            else:
                x1 = img_loader(self.basedir, path_x + file_ending, self.n_channels)
                if path_x == path_xd: 
                    x2 = x1.copy()
                else: 
                    x2 = img_loader(self.basedir, path_xd + file_ending, self.n_channels)

        # If an HF processor is provided, produce pixel_values here
        if self.processor is not None:
            inputs = self.processor(images=[x1, x2], return_tensors='pt')
            pv = inputs['pixel_values']  # shape: (2, C, H, W)
            x1, x2 = pv[0], pv[1]

        elif self.transform is not None and not self.is_latent:
            # Only apply transform if not using cached latent data
            x1 = self.transform(x1) # cached and precomputed 
            if not self.distill: 
                x2 = self.transform(x2)

        # apply augmentation at train/validation time
        if self.aug_transform is not None: 
            # Apply augmentation
            x1 = self.transform(x1) # cached and precomputed 
            if not self.distill: 
                x2 = self.aug_transform(x2)

        if self.phase == 'TEST' or self.fitmode: 
            label = torch.tensor(label)
            return x1, x2, label, label, label # for backwards compatabilty 
        return x1, x2


def pil_loader(base_dir, path, n_channels):
    img = Image.open(os.path.join(base_dir, path)).convert('RGB')
    return img

def img_loader(base_dir, path, n_channels): 
    if path.endswith(".pt"): 
        return torch.load(os.path.join(base_dir, path))
    else: 
        return pil_loader(base_dir, path, n_channels)
         

class SiameseDataset(data.Dataset):
    """ For testing purposes to compare with previous work"""
    def __init__(self, basedir="./", phase='training', n_channels=3, transform=None,
                 image_path='./', save_path=None, train_file="", val_file="", test_file="", processor=None):

        self.phase = phase
        self.basedir = basedir
        self.processor = processor


        if self.phase == 'training':
            # In this way, images from one patient only appear in one subset.
            print(f"Loading training images from {train_file}")
            self.image_pairs = np.loadtxt(train_file, dtype=str)
        elif self.phase == 'validation':
            self.image_pairs = np.loadtxt(val_file, dtype=str)
        elif self.phase == 'testing':
            self.image_pairs = np.loadtxt(test_file, dtype=str)
        else:
            raise Exception('Invalid argument for parameter phase!')
        self.n_samples = len(self.image_pairs)

        self.n_channels = n_channels
        self.transform = transform

        # deprecated
        self.PATH = image_path 

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):

        x1 = pil_loader(self.basedir, self.image_pairs[index][0], self.n_channels)
        x2 = pil_loader(self.basedir, self.image_pairs[index][1], self.n_channels)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        # If an HF processor is provided, produce pixel_values here
        if self.processor is not None:
            to_pil = transforms.ToPILImage()
            img1 = x1 if isinstance(x1, Image.Image) else to_pil(x1)
            img2 = x2 if isinstance(x2, Image.Image) else to_pil(x2)
            inputs = self.processor(images=[img1, img2], return_tensors='pt')
            pv = inputs['pixel_values']
            x1, x2 = pv[0], pv[1]

        y1 = float(self.image_pairs[index][2])

        return x1, x2, 0, 0, y1
