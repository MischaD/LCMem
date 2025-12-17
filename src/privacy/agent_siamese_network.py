from builtins import NotImplementedError
import time
import copy
import json
import torch
import torchio as tio
import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torchvision.transforms import functional as TF
from PIL import Image
from io import BytesIO
import numpy as np
try:
    from transformers import AutoImageProcessor
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# Optional W&B support
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except Exception:
    _wandb = None
    _WANDB_AVAILABLE = False


class RandomChannelNormalization:
    """
    Randomly applies channel-wise normalization to individual samples using their own statistics.
    Similar to the VAE normalization but computed per-sample instead of using global statistics.
    """
    def __init__(self, probability=0.5, final_mean=0.0, final_std=0.5):
        self.probability = probability
        self.final_mean = final_mean
        self.final_std = final_std
    
    def __call__(self, x):
        if torch.rand(1).item() < self.probability:
            return self._normalize_sample(x)
        return x
    
    def _normalize_sample(self, x):
        """
        Apply channel-wise normalization to a single sample using its own statistics.
        """
        # Compute statistics for each channel across spatial dimensions
        channel_mean = x.mean(dim=(2, 3), keepdim=True)  # (C, 1, 1, 1)
        channel_std = x.std(dim=(2, 3), keepdim=True)  # (C, 1, 1, 1)
        
        # Avoid division by zero
        channel_std = torch.clamp(channel_std, min=1e-8)
        
        x_normalized = (x - channel_mean) / channel_std
        x_normalized = x_normalized * self.final_std + self.final_mean
        
        return x_normalized


def jpeg_compress_transform(quality_range):
    """Create a JPEG compression transform."""
    def _fn(x):
        if torch.rand(1).item() < 0.5:  # Apply with 50% probability
            quality = torch.rand(1).item() * (quality_range[1] - quality_range[0]) + quality_range[0]
            pil = TF.to_pil_image(x)
            buf = BytesIO()
            pil.save(buf, format='JPEG', quality=int(quality))
            buf.seek(0)
            out = Image.open(buf).convert('RGB')
            return TF.to_tensor(out)
        return x
    return _fn


from dataset.utils import Utils
import dataset.utils.fc_model as ufc 
from dataset.utils.EarlyStopping import EarlyStopping
from networks.SiameseNetwork import SiameseNetwork
from sklearn import metrics
import torch.distributed as dist
import os


class WBLogger:
    """Safe W&B wrapper. Works even if wandb is not installed or no run is passed."""
    def __init__(self, run=None):
        self.run = run

    def log(self, data, step=None, commit=None):
        if self.run:
            # keep kwargs only if provided to avoid API differences
            kwargs = {}
            if step is not None: kwargs["step"] = step
            if commit is not None: kwargs["commit"] = commit
            self.run.log(data, **kwargs)

    def watch(self, model):
        if self.run:
            self.run.watch(model)

    def summary_update(self, **items):
        if self.run:
            for k, v in items.items():
                self.run.summary[k] = v

    @property
    def name(self):
        return self.run.name if self.run else None


class AgentSiameseNetwork:
    def __init__(self, config, wb_run=None):
        self.config = config
        self.wb = WBLogger(wb_run)
        
        # Distributed training setup
        self.distributed = self.config.get('distributed', False)
        
        # Get local rank from environment variable (set by torchrun) or config
        if self.distributed:
            import os
            self.local_rank = int(os.environ.get('LOCAL_RANK', self.config.get('local_rank', 0)))
            
            # Verify CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but distributed training was requested")
            
            # Check if the requested GPU exists
            if self.local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"Local rank {self.local_rank} exceeds available GPU count {torch.cuda.device_count()}")
            
            # Set device before initializing process group
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
            
            # Initialize process group
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl')
            print(f"Distributed training enabled on GPU {self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Single GPU training on device: {self.device}")

        # set path used to save experiment-related files and results
        self.SAVINGS_PATH = './archive/' + self.config['experiment_description'] + '/'
        self.IMAGE_PATH = self.config['image_path']
        self.filelist = self.config['filelist']
        self.basedir = self.config['basedir']

        # save configuration as config.json in the created folder
        with open(self.SAVINGS_PATH + 'config.json', 'w') as outfile:
            json.dump(self.config, outfile, indent='\t')
            outfile.close()

        # enable benchmark mode in cuDNN
        torch.backends.cudnn.benchmark = True

        # set all the important variables
        self.network = self.config['siamese_architecture']

        self.num_workers = self.config['num_workers']
        self.pin_memory = self.config['pin_memory']

        #layer_to_in_channels = {0: 3, 1: 96, 2:192, 3:384, 4: 768, 5:768}
        self.n_channels = self.config["n_channels"]#layer_to_in_channels[self.config["layer"]] if self.config["layer"] != -1 else 3

        self.input_type = self.config['input_type']
        self.n_features = self.config['n_features']
        self.image_size = self.config['image_size']
        self.loss_method = self.config['loss']
        self.optimizer_method = self.config['optimizer']
        self.learning_rate = float(self.config['learning_rate'])
        self.alpha = float(self.config.get('alpha', 0.5))
        self.batch_size = self.config['batch_size']
        self.max_epochs = self.config['max_epochs']
        self.early_stopping = self.config['early_stopping']
        self.transform = self.config['transform']
        self.do_test = self.config['do_test'] if hasattr(config, 'do_test') else True

        # Sample counts will be set dynamically based on dataset lengths
        self.n_samples_train = None
        self.n_samples_val = None
        self.n_samples_test = None
        
        # Inpainting configuration

        self.start_epoch = 0

        self.es = EarlyStopping(patience=self.early_stopping)
        self.best_loss = 100000
        self.loss_dict = {'training': [],
                          'validation': []}

        self.balanced = True
        self.randomized = False

        # define the suffix needed for loading the checkpoint (in case you want to resume a previous experiment)
        if self.config['resumption'] is True:
            if self.config['resumption_count'] == 1:
                self.load_suffix = ''
            elif self.config['resumption_count'] == 2:
                self.load_suffix = '_resume'
            elif self.config['resumption_count'] > 2:
                self.load_suffix = '_resume' + str(self.config['resumption_count'] - 1)

        # define the suffix needed for saving the checkpoint (the checkpoint is saved at the end of each epoch)
        if self.config['resumption'] is False:
            self.save_suffix = ''
        elif self.config['resumption'] is True:
            if self.config['resumption_count'] == 1:
                self.save_suffix = '_resume'
            elif self.config['resumption_count'] > 1:
                self.save_suffix = '_resume' + str(self.config['resumption_count'])


        # Define the siamese neural network architecture
        backbone_cfg = None
        if self.network == 'dinov3_hf':
            backbone_cfg = {
                'model_id': self.config.get('dinov3_model_id', 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'),
                'dino_layers': self.config.get('dino_layer', 2),
                'out_method': self.config.get('dino_out_method', 'cls_only'),
                'dropout': self.config.get('dropout', 0.0),
                'finetune_backbone': self.config.get('finetune_backbone', False),
                'sigmoid_out': self.config.get('sigmoid_out', False),
            }
        else: 
            backbone_cfg = {
                'dropout': self.config.get('dropout', 0.0),
                'finetune_backbone': self.config.get('finetune_backbone', False),
            }

        # Initialize models on device
        # Check if we should use ImageSpaceSiameseNetwork (for on-the-fly VAE encoding)
        use_image_space = self.config.get('use_image_space_network', False)
        self.use_distill = self.config.get("distill", False)
        self.alpha_distill = self.config.get('alpha_distill', 1.0)
        
        if use_image_space:
            from networks.SiameseNetwork import ImageSpaceSiameseNetwork
            vae_model_path = self.config.get('vae_model_path', 'stabilityai/stable-diffusion-2')
            final_mean = self.config.get('final_mean', 0.0)
            final_std = self.config.get('final_std', 0.5)
            
            self.net = ImageSpaceSiameseNetwork(
                network=self.network,
                in_channels=self.n_channels,
                n_features=self.n_features,
                backbone_cfg=backbone_cfg,
                vae_model_path=vae_model_path,
                final_mean=final_mean,
                final_std=final_std,
                distill=self.use_distill
            ).to(self.device)
            self.best_net = ImageSpaceSiameseNetwork(
                network=self.network,
                in_channels=self.n_channels,
                n_features=self.n_features,
                backbone_cfg=backbone_cfg,
                vae_model_path=vae_model_path,
                final_mean=final_mean,
                final_std=final_std,
                distill=self.use_distill
            ).to(self.device)
            print(f"Using ImageSpaceSiameseNetwork with VAE: {vae_model_path}")
        else:
            self.net = SiameseNetwork(network=self.network, in_channels=self.n_channels, n_features=self.n_features, backbone_cfg=backbone_cfg).to(self.device)
            self.best_net = SiameseNetwork(network=self.network, in_channels=self.n_channels,
                                           n_features=self.n_features, backbone_cfg=backbone_cfg).to(self.device)
        
        # Apply DistributedDataParallel wrapper for distributed training
        if self.distributed:
            self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=[self.local_rank])
            self.best_net = nn.parallel.DistributedDataParallel(self.best_net, device_ids=[self.local_rank])
            print(f"Models wrapped with DistributedDataParallel on GPU {self.local_rank}")
        # Get n_features from the underlying model (handle DistributedDataParallel wrapper)
        if hasattr(self.net, 'module'):
            self.n_features = self.net.module.n_features
        else:
            self.n_features = self.net.n_features

        self.wb.watch(self.net)

        # Print a concise model summary
        self._print_model_summary(backbone_cfg)
        
        # Load pretrained model if path is specified
        if self.config.get('model_path'):
            print(f"Loading model path: {self.config['model_path']}")
            state_dict = torch.load(self.config['model_path'], map_location=self.device)
            if state_dict.get("state_dict", None) is not None: 
                state_dict = state_dict["state_dict"]
            
            # Handle loading into DistributedDataParallel models
            if hasattr(self.net, 'module'):
                self.net.module.load_state_dict(state_dict)
                self.best_net.module.load_state_dict(state_dict)
            else:
                if use_image_space:
                    self.net.load_state_dict(state_dict, strict=False)
                    self.best_net.load_state_dict(state_dict, strict=False)
                else: 
                    self.net.load_state_dict(state_dict)
                    self.best_net.load_state_dict(state_dict)

        # Choose loss function
        if self.loss_method == 'BCEWithLogitsLoss':
            self.loss = nn.BCEWithLogitsLoss().to(self.device)
        else:
            raise Exception('Invalid argument: ' + self.loss_method +
                            '\nChoose BCEWithLogitsLoss! Other loss functions are not yet implemented!')

        # Set the optimizer function
        if self.optimizer_method == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        else:
            raise Exception('Invalid argument: ' + self.optimizer_method +
                            '\nChoose Adam! Other optimizer functions are not yet implemented!')

        # Set up cosine learning rate scheduler
        eta_min=self.learning_rate * self.config["eta_min"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epochs, eta_min=eta_min)
        print(f'Cosine learning rate scheduler initialized: T_max={self.max_epochs}, eta_min={eta_min:.6f}')

        # load state dicts and other information in case a previous experiment will be continued
        if self.config['resumption'] is True:
            self.checkpoint = torch.load('./archive/' + self.config['previous_experiment'] + '/' + self.config[
                'previous_experiment'] + '_checkpoint' + self.load_suffix + '.pth', map_location=self.device)
            
            best_net_state = torch.load(
                './archive/' + self.config['previous_experiment'] + '/' + self.config[
                    'previous_experiment'] + '_best_network' + self.load_suffix + '.pth', map_location=self.device)
            
            # Handle loading into DistributedDataParallel models
            if hasattr(self.best_net, 'module'):
                self.best_net.module.load_state_dict(best_net_state)
                self.net.module.load_state_dict(self.checkpoint['state_dict'])
            else:
                self.best_net.load_state_dict(best_net_state)
                self.net.load_state_dict(self.checkpoint['state_dict'])
                
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            # Load scheduler state if it exists
            if 'scheduler' in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            self.best_loss = self.checkpoint['best_loss']
            self.loss_dict = self.checkpoint['loss_dict']
            self.es.best = self.checkpoint['best_loss']
            self.es.num_bad_epochs = self.checkpoint['num_bad_epochs']
            self.start_epoch = self.checkpoint['epoch']

        # Handle resume from specific checkpoint file
        elif self.config.get('resume_checkpoint') is not None:
            checkpoint_path = self.config['resume_checkpoint']
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            
            import os
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if hasattr(self.net, 'module'):
                try: 
                    self.net.module.load_state_dict(self.checkpoint['state_dict'])
                except RuntimeError: 
                    from collections import OrderedDict
                    state_dict = OrderedDict()
                    for k in self.checkpoint['state_dict']:
                        k_out = k[7:]
                        state_dict[k_out] = self.checkpoint['state_dict'][k]
                    self.net.module.load_state_dict(state_dict)
                
            else:
                try: 
                    self.net.load_state_dict(self.checkpoint['state_dict'])
                except RuntimeError: 
                    from collections import OrderedDict
                    state_dict = OrderedDict()
                    for k in self.checkpoint['state_dict']:
                        k_out = k[7:]
                        state_dict[k_out] = self.checkpoint['state_dict'][k]
                    self.net.load_state_dict(state_dict)
            
                
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            
            # Load scheduler state if it exists
            if 'scheduler' in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            
            self.best_loss = self.checkpoint['best_loss']
            self.loss_dict = self.checkpoint['loss_dict']
            self.es.best = self.checkpoint['best_loss']
            self.es.num_bad_epochs = self.checkpoint['num_bad_epochs']
            self.start_epoch = self.checkpoint['epoch']
            
            print(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.6f}")
            
            # Also load the best model if it exists in the same directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            # Replace '_checkpoint' with '_best_network' to find the best model
            best_model_path = os.path.join(checkpoint_dir, checkpoint_name.replace('_checkpoint', '_best_network') + '.pth')
            
            if os.path.exists(best_model_path):
                best_net_state = torch.load(best_model_path, map_location=self.device)
                if hasattr(self.best_net, 'module'):
                    self.best_net.module.load_state_dict(best_net_state)
                else:
                    self.best_net.load_state_dict(best_net_state)
                print(f"Loaded best model from: {best_model_path}")
            else:
                print(f"Best model not found at {best_model_path}, using current model as best")

        # Initialize transformations
        if self.network == 'dinov3_hf':
            # Use light transforms; AutoImageProcessor will handle resize/normalize
            self.transform_train = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
            self.transform_val_test = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
            # Prepare HF processor for dataset-level preprocessing
            if not _HAS_TRANSFORMERS:
                raise ImportError("transformers not installed; required for dinov3_hf")
            self.hf_processor = AutoImageProcessor.from_pretrained(backbone_cfg['model_id'])
            # Ensure processor size matches configured image size
            self.hf_processor.size = {"height": self.image_size, "width": self.image_size}
        elif self.transform == 'image_net':
            self.transform_train = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.transform_val_test = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.transform == "default":
            self.transform_train = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])
            self.transform_val_test = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])

        elif self.transform == 'none': 
            self.transform_train = lambda x: x
            self.transform_val_test = lambda x: x
        

        if self.input_type != "latent": 
            trafos = [lambda x: x.unsqueeze(1) if x.ndim == 3 else x,]
            if self.config["aug"]["rotate"]: 
                angle = self.config["aug"]["rotate"]
                trafos.append(
                    tio.RandomAffine( degrees=(-angle,angle,0,0,0,0), scales = 0, default_pad_value = 'minimum',p =0.5)
                )
            if self.config["aug"]["flip"]: 
                trafos.append(
                    tio.RandomFlip(axes=(2), flip_probability=0.5)
                )
            if self.config["aug"]["blur"]:
                trafos.append(
                    tio.RandomBlur(std=self.config["aug"]["blur"])#2.5
                )
            if self.config["aug"]["noise"]:
                trafos.append(
                    tio.RandomNoise(std=self.config["aug"]["noise"])#0.15
                )
            if hasattr(self.config["aug"], "zoom") and self.config["aug"]["zoom"]:
                p = self.config["aug"]["zoom"]
                trafos.append(
                    transforms.RandomApply(
                        [transforms.RandomResizedCrop(size=self.input_size, scale=(0.65, 0.8), ratio=(1,1)),], p=p
                    )
                )

            trafos += [lambda x: torch.clamp(x, 0.0, 1.0), lambda x: x.squeeze(1) if x.ndim == 4 else x]
            
            # Add torchvision transforms after squeeze (for [C,H,W] tensors)
            torchvision_trafos = []
            if self.config["aug"].get("brightness"):
                factor = self.config["aug"]["brightness"]
                torchvision_trafos.append(
                    transforms.RandomApply([transforms.ColorJitter(brightness=(float(factor), float(factor)))], p=0.5)
                )
            if self.config["aug"].get("contrast"):
                factor = self.config["aug"]["contrast"]
                torchvision_trafos.append(
                    transforms.RandomApply([transforms.ColorJitter(contrast=(float(factor), float(factor)))], p=0.5)
                )
            if self.config["aug"].get("jpeg_compression"):
                quality_range = self.config["aug"]["jpeg_compression"]
                torchvision_trafos.append(
                    transforms.Lambda(jpeg_compress_transform(quality_range))
                )
            if torchvision_trafos:
                torchvision_trafos.append(lambda x: torch.clamp(x, 0.0, 1.0))
                torchvision_compose = transforms.Compose(torchvision_trafos)
                trafos.append(lambda x: torchvision_compose(x) if x.ndim == 3 else x)
            
            aug_transform = tio.Compose(trafos)
        else: 
            trafos = [lambda x: x.unsqueeze(1) if x.ndim == 3 else x,] # tio expects 3d
            if self.config["aug"]["noise"]:
                trafos.append(
                    tio.RandomNoise(std=self.config["aug"]["noise"])#0.15
                )
            # Add random normalization if specified
            if self.config["aug"].get("randomlyapplynormalization", 0.0) > 0.0:
                norm_prob = self.config["aug"]["randomlyapplynormalization"]
                trafos.append(
                    RandomChannelNormalization(probability=norm_prob)
                )
            
            trafos.append(lambda x: x.squeeze(1))
            aug_transform = tio.Compose(trafos)

        self.mini_epoch_size = self.config['mini_epoch_size']
        if self.input_type == "image" or  self.input_type == "latent": 
            self.train_ds = Utils.get_data_sets(
                basedir=self.basedir,
                phase='TRAIN',
                n_channels=self.n_channels,
                transform=self.transform_train, 
                filelist=self.filelist,
                aug_transform=aug_transform,
                processor=getattr(self, 'hf_processor', None),
                unsupervised=self.config['unsupervised'], 
                is_latent = self.input_type == "latent",
                distill=self.use_distill
            )

            self.val_ds = Utils.get_data_sets(
                basedir=self.basedir,
                phase='VAL',
                n_channels=self.n_channels,
                transform=self.transform_val_test, 
                filelist=self.filelist,
                #aug_transform=aug_transform,
                processor=getattr(self, 'hf_processor', None),
                unsupervised=self.config['unsupervised'],
                is_latent = self.input_type == "latent",
                distill=False#,self.use_distill
            )

            if self.use_distill: 
                print("Computing distillation")
                self.train_ds.precopute_for_distillation(self.transform_val_test, self.net)
                #self.val_ds.precopute_for_distillation(self.transform_val_test, self.net)


                # Check if we should use image pairs for testing
            if self.config.get('image_pairs_test') is not None:
                # Use SiameseDataset with image pairs for testing
                self.test_ds = Utils.get_siamese_data_sets(
                    basedir=self.basedir,
                    phase='testing',
                    n_channels=self.n_channels,
                    transform=self.transform_val_test,
                    image_path=self.basedir,
                    test_file=self.config['image_pairs_test'],
                    processor=getattr(self, 'hf_processor', None),

                )
                self.test_loader = torch.utils.data.DataLoader(
                    self.test_ds, 
                    batch_size=self.batch_size, 
                    shuffle=False, 
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory

                )
            else:
                self.test_loader = Utils.get_data_loaders(phase='TEST', 
                                                        n_channels=self.n_channels,
                                                        transform=self.transform_val_test, basedir=self.basedir, filelist=self.filelist,
                                                        batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                        processor=getattr(self, 'hf_processor', None),
                                                        is_latent = self.input_type == "latent",
                                                        )
        else: 
            raise NotImplementedError("input type not supported")



        self.n_samples_train = len(self.train_ds)
        self.n_samples_val = len(self.val_ds)
        self.n_samples_test = len(self.test_loader.dataset)
        print(f"Dataset sizes (mini-epoch mode) - Train: {self.n_samples_train}, Val: {self.n_samples_val}, Test: {self.n_samples_test}")


    def _broadcast_indices(self, idx, device):
        # idx is a 1D LongTensor on rank 0
        if dist.get_rank() == 0:
            num = torch.tensor([idx.numel()], dtype=torch.int64, device=device)
        else:
            num = torch.empty(1, dtype=torch.int64, device=device)
        dist.broadcast(num, src=0)
        if dist.get_rank() != 0:
            idx = torch.empty(num.item(), dtype=torch.int64, device=device)
        dist.broadcast(idx, src=0)
        return idx.cpu()  # DataLoader expects CPU tensors for Subset

    def training_validation(self):
        for epoch in tqdm.tqdm(range(self.start_epoch, self.max_epochs), "Training"):
            start_time = time.time()

            if self.mini_epoch_size:
                device = torch.device(f"cuda:{self.local_rank}") if self.distributed else torch.device("cpu")

                if not self.distributed or dist.get_rank() == 0:
                    tr_idx = torch.randperm(len(self.train_ds), device=device)[: self.mini_epoch_size]
                    va_idx = torch.randperm(len(self.val_ds), device=device)[: max(1, self.mini_epoch_size // 4)]
                else:
                    tr_idx = torch.tensor([], dtype=torch.int64, device=device)
                    va_idx = torch.tensor([], dtype=torch.int64, device=device)

                if self.distributed:
                    tr_idx = self._broadcast_indices(tr_idx, device)
                    va_idx = self._broadcast_indices(va_idx, device)


                subset_train = torch.utils.data.Subset(self.train_ds, tr_idx)
                subset_val   = torch.utils.data.Subset(self.val_ds, va_idx)

                train_sampler = None
                if self.distributed:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        subset_train, shuffle=True, drop_last=True
                    )

                self.training_loader = torch.utils.data.DataLoader(
                    subset_train,
                    batch_size=self.batch_size,
                    shuffle=(train_sampler is None),
                    num_workers=self.num_workers,
                    sampler=train_sampler,
                    pin_memory=self.pin_memory,
                    persistent_workers=False,
                    prefetch_factor=4,
                )

                if self.distributed and hasattr(self.training_loader.sampler, "set_epoch"):
                    self.training_loader.sampler.set_epoch(epoch)

                val_sampler = None
                if self.distributed:
                    val_sampler = torch.utils.data.distributed.DistributedSampler(
                        subset_val, shuffle=False, drop_last=True
                    )

                self.validation_loader = torch.utils.data.DataLoader(
                    subset_val,
                    batch_size=self.batch_size,
                    shuffle=(val_sampler is None),
                    num_workers=self.num_workers,
                    sampler=val_sampler,
                    pin_memory=self.pin_memory,
                    persistent_workers=False,
                    prefetch_factor=4,
                )

                self.n_samples_train = len(subset_train)
                self.n_samples_val = len(subset_val)

            n_train = len(train_sampler) if train_sampler is not None else self.n_samples_train
            training_loss, train_nce, train_fc = Utils.train(self.net, self.training_loader, n_train, self.batch_size,
                                        self.loss, self.optimizer, epoch, self.max_epochs, alpha=self.alpha, distill=self.use_distill, alpha_distill=self.alpha_distill)

            #self.validation_loader.dataset.reset_generator() # make sure this is deterministic for more accurate loss, useless for miniepoch
            n_val = len(val_sampler) if val_sampler is not None else self.n_samples_val
            validation_loss, val_nce, val_fc = Utils.validate(self.net, self.validation_loader, n_val, self.batch_size,
                                            self.loss, epoch, self.max_epochs, alpha=self.alpha)
 
            if self.distributed:
                t = torch.tensor([validation_loss], device=f"cuda:{self.local_rank}")
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                validation_loss = (t.item() / torch.distributed.get_world_size())

            if not self.distributed or self.local_rank == 0:
                if validation_loss < self.best_loss:
                    self.best_loss = validation_loss
                    self.best_net = copy.deepcopy(self.net)

                self.loss_dict['training'].append(training_loss)
                self.loss_dict['validation'].append(validation_loss)

                end_time = time.time()
                print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                    round((end_time - start_time) / 60, 2)) + ' minutes')

                # Save best model (unwrap DistributedDataParallel if needed)
                best_model_state = self.best_net.state_dict()
                if hasattr(self.best_net, 'module'):
                    best_model_state = self.best_net.module.state_dict()
            
            # Step the learning rate scheduler
            self.scheduler.step()

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Only save on main process for distributed training
            if not self.distributed or self.local_rank == 0:
                torch.save(best_model_state, self.SAVINGS_PATH + self.config[
                    'experiment_description'] + '_best_network' + self.save_suffix + '.pth')

                Utils.save_loss_curves(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])
                Utils.plot_loss_curves(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])

                print(f'Current learning rate: {current_lr:.6f}')
                self.wb.log({
                    "epoch": epoch + 1,
                    "loss/train": float(training_loss),
                    "loss/val": float(validation_loss),
                    "loss_nce/train": float(train_nce),
                    "loss_fc/train": float(train_fc),
                    "loss_nce/val": float(val_nce),
                    "loss_fc/val": float(val_fc),
                    "lr": float(current_lr),
                })
                Utils.save_checkpoint(epoch, self.net, self.optimizer, self.scheduler, self.loss_dict, self.best_loss,
                                      self.es.num_bad_epochs, self.SAVINGS_PATH + self.config[
                                          'experiment_description'] + '_checkpoint' + self.save_suffix + '.pth')

            if self.distributed:
                dist.barrier()

            es_check = self.es.step(validation_loss)
            if es_check:
                break

        print('Finished Training!')
        
        # Clean up distributed training
        if self.distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _print_model_summary(self, backbone_cfg=None):

        if not self.distributed or self.local_rank == 0:
            print("\nModel Summary")
            print("============")
            print(f"Architecture: {self.network}")
            if self.network == 'dinov3_hf' and backbone_cfg is not None:
                print(f"Backbone: {backbone_cfg.get('model_id')}")
                print(f"Feature: {backbone_cfg.get('feature')} (backbone frozen)")
            
            # Handle DistributedDataParallel model printing
            model_to_print = self.net
            if hasattr(self.net, 'module'):
                model_to_print = self.net.module
                
            print(model_to_print if self.network != 'dinov3_hf' else model_to_print.fc_end)

            total_params = sum(p.numel() for p in self.net.parameters())
            trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            if trainable_params < total_params:
                print("Trainable parameter groups:")
                for name, p in self.net.named_parameters():
                    if p.requires_grad:
                        print(f"  - {name}: {p.numel():,}")
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_params:,}")
            print(f"Device: {self.device}")
            if self.distributed:
                print(f"Distributed training on GPU {self.local_rank}")

    
    def testing_evaluation(self):
        # Testing phase!
        #self.test_loader.dataset.reset_generator()
        if not self.distributed or self.local_rank == 0:
            if len(self.test_loader.dataset) == 0:
                print("No testdata added. Skipping testing (Probably no images with same ID)")
                return 1.0  

            if self.config["nce"]:  
                y_true, y_pred = Utils.test(self.best_net, self.test_loader)
            else: 
                y_true, y_pred = ufc.test(self.best_net, self.test_loader)
            y_true, y_pred = [y_true.numpy(), y_pred.numpy()]

            # Compute the evaluation metrics!
            fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.roc_auc_score(y_true, y_pred)
            y_pred_thresh = Utils.apply_threshold(y_pred, 0.5)
            accuracy, f1_score, precision, recall, report, confusion_matrix = Utils.get_evaluation_metrics(y_true,
                                                                                                        y_pred_thresh)
            auc_mean, confidence_lower, confidence_upper = Utils.bootstrap(10000,
                                                                        y_true,
                                                                        y_pred,
                                                                        self.SAVINGS_PATH,
                                                                        self.config['experiment_description'])

            # Plot ROC curve!
            Utils.plot_roc_curve(fp_rates, tp_rates, self.SAVINGS_PATH, self.config['experiment_description'])

            # Save all the results to files!
            Utils.save_labels_predictions(y_true, y_pred, y_pred_thresh, self.SAVINGS_PATH,
                                        self.config['experiment_description'])

            Utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix,
                                    self.SAVINGS_PATH, self.config['experiment_description'])

            Utils.save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, self.SAVINGS_PATH,
                                        self.config['experiment_description'])

            # Print the evaluation metrics!
            print('EVALUATION METRICS:')
            print('AUC: ' + str(auc))
            print('Accuracy: ' + str(accuracy))
            print('F1-Score: ' + str(f1_score))
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
            print('Report: ' + str(report))
            print('Confusion matrix: ' + str(confusion_matrix))

            print('BOOTSTRAPPING: ')
            print('AUC Mean: ' + str(auc_mean))
            print('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))

            self.wb.log({
                "test/auc": float(auc),
                "test/accuracy": float(accuracy),
                "test/f1": float(f1_score),
                "test/precision": float(precision),
                "test/recall": float(recall),
                "test/auc_mean_bootstrap": float(auc_mean),
            })
            self.wb.summary_update(final_recall=float(recall))

            return recall
    
    def run(self):
        # Call training/validation and testing loop successively
        self.training_validation()
        if self.do_test: 
            test_score = self.testing_evaluation()
            return test_score
        else: 
            return self.best_loss

