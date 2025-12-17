import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
try:
    # Optional dependency; only used when network == 'dinov3_hf'
    from transformers import AutoImageProcessor, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class ToTensorIfNotTensor:
    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            return input
        return F.to_tensor(input)

class LambdaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

    def forward_once(self, x):
        return x



class SiameseNetwork(nn.Module):
    def __init__(self, network='ResNet-50', in_channels=3, n_features=128, backbone_cfg=None):
        super(SiameseNetwork, self).__init__()
        self.network = network
        self.in_channels = in_channels
        self.n_features = n_features
        self.backbone_cfg = backbone_cfg if backbone_cfg is not None else {"dropout": 0.0, "finetune_backbone": True}
        self.dropout = self.backbone_cfg["dropout"]
        self.finetune_backbone = self.backbone_cfg["finetune_backbone"]

        if self.network in ['ResNet-50', 'ResNet-101']:
            # Model: Use ResNet-50 architecture
            if self.network == 'ResNet-50': 
                self.model = models.resnet50(pretrained=True)
            else: 
                self.model = models.resnet101(pretrained=True)
            # Adjust the input layer: either 1 or 3 input channels
            if self.in_channels != 3:
                print(f"Overwriting first Conv layer with in_channels={in_channels}")
                self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            

            # Adjust the ResNet classification layer to produce feature vectors of a specific size
            self.model.fc = nn.Linear(in_features=2048, out_features=self.n_features, bias=True)
            self.fc_end = nn.Linear(self.n_features, 1)
            self.transform_lazy = None
            self.model = self.model.to("cuda")

        elif self.network == 'dinov3_hf':
            if not _HAS_TRANSFORMERS:
                raise ImportError("transformers not installed; required for dinov3_hf")

            from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer
            # Load HF DINOv3 backbone and processor
            model_id = self.backbone_cfg.get('model_id', 'facebook/dinov3-vits16-pretrain-lvd1689m')
            self.dino_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.processor.size = {
                "height": 512,
                "width": 512
            }
            # Ensure hidden states are available
            self.backbone = AutoModel.from_pretrained(model_id).to(self.dino_device).eval()
            for p in self.backbone.parameters():
                p.requires_grad = self.finetune_backbone 

            dino_cfg = self.backbone.config
            self.rope_embedding = self.backbone.rope_embeddings(torch.zeros(1, 1, self.processor.size["height"], self.processor.size["width"]).to(self.dino_device))
            self.out_layers = nn.ModuleList([DINOv3ViTLayer(dino_cfg) for _ in range(self.backbone_cfg["dino_layers"])])
            for p in self.out_layers.parameters():
                p.requires_grad = True 
            
            if self.backbone_cfg.get('out_method') == "cls_only":
                self.fc_end = nn.Linear(dino_cfg.hidden_size, 1)
            else: 
                self.fc_end = nn.Linear(2*dino_cfg.hidden_size, 1)
            # Lazy-init head once feature size is known at first forward

        elif self.network.startswith("bfid_"): 
            model_name = self.network[5:]
            from beyondfid.feature_extractor_models import load_feature_model, _FEATURE_MODELS
            from beyondfid.default_config import config
            config = config.feature_extractors.get(model_name)
            if config is None:
                raise ValueError(f"Model: {model_name} not found. Available models: {list(_FEATURE_MODELS.keys())}")

            if model_name == "byol": 
                config.config.model_path = "/vol/ideadata/ed52egek/pycharm/improve_recall/BeyondFID/beyondfid/feature_extractor_models/byol/large_model.pth"
                config.config.cfg_path = "/vol/ideadata/ed52egek/pycharm/improve_recall/BeyondFID/beyondfid/feature_extractor_models/byol/config_large.yaml"

            self.model = load_feature_model(config).to("cuda")

            out_features = self.model(torch.zeros((1, 3, 512, 512)).to("cuda")).size()[-1]
            self.fc_end = nn.Linear(out_features, 1).to("cuda")
            self.transform_lazy = None
            self.model = self.model.to("cuda")
            if not self.finetune_backbone: 
                for p in self.model.parameters():
                    p.requires_grad = False

        elif self.network in ['ConvNeXt-Tiny', 'ConvNeXt-Small', 'ConvNeXt-Base', 'ConvNeXt-Large']:
            # Model: Use ConvNeXt architecture
            if self.network == 'ConvNeXt-Tiny':
                self.model = models.convnext_tiny(pretrained=True)
                feature_dim = 768
            elif self.network == 'ConvNeXt-Small':
                self.model = models.convnext_small(pretrained=True)
                feature_dim = 768
            elif self.network == 'ConvNeXt-Base':
                self.model = models.convnext_base(pretrained=True)
                feature_dim = 1024
            elif self.network == 'ConvNeXt-Large':
                self.model = models.convnext_large(pretrained=True)
                feature_dim = 1536
            
            # Adjust the input layer for different channel counts
            if self.in_channels != 3:
                print(f"Overwriting first Conv layer with in_channels={self.in_channels}")
                # ConvNeXt uses a different first layer structure
                if hasattr(self.model.features, '0'):
                    # For ConvNeXt, the first layer is in features.0
                    original_conv = self.model.features[0][0]  # Get the Conv2d layer
                    self.model.features[0][0] = nn.Conv2d(
                        self.in_channels, 
                        original_conv.out_channels, 
                        kernel_size=original_conv.kernel_size, 
                        stride=original_conv.stride, 
                        padding=original_conv.padding, 
                        bias=original_conv.bias is not None
                    )
            
            # Adjust the classification layer to produce feature vectors of a specific size
            self.model.classifier[2] = nn.Linear(in_features=feature_dim, out_features=self.n_features, bias=True)
            self.fc_end = nn.Linear(self.n_features, 1)
            self.transform_lazy = None
            self.model = self.model.to("cuda")
            
            # Handle finetune_backbone setting
            if not self.finetune_backbone:
                for p in self.model.parameters():
                    p.requires_grad = False
                # Only train the classifier layers
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
                for p in self.fc_end.parameters():
                    p.requires_grad = True

        else:
            raise Exception('Invalid argument: ' + self.network +
                            '\nChoose ResNet-50, ResNet-101, ConvNeXt-Tiny, ConvNeXt-Small, ConvNeXt-Base, ConvNeXt-Large, or dinov3_hf! Other architectures are not yet implemented in this framework.')


    def forward_once(self, x):
        # Forward function for one branch to get the n_features-dim feature vector before merging
        if self.network == 'dinov3_hf':
            # Expect dataset to provide preprocessed pixel_values; also accept raw tensor
            device = self.dino_device
            if isinstance(x, dict):
                pixel_values = x.get('pixel_values', None)
                if pixel_values is None:
                    raise ValueError("Expected 'pixel_values' in input dict for dinov3_hf.")
            else:
                pixel_values = x
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.to(device)
            if not self.finetune_backbone: 
                with torch.no_grad():
                    out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
                    hidden_states = out.last_hidden_state
            else: 
                out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
                hidden_states = out.last_hidden_state


            # Apply dropout to hidden states
            hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            cls_token = hidden_states[:, 0, :] # frozen pooler_output

            for i, layer_module in enumerate(self.out_layers): 
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=None, 
                    position_embeddings=self.rope_embedding,
                )
            
            if self.backbone_cfg.get('out_method') != "cls_only" and self.out_layers != 0: 
                output = torch.cat([cls_token, hidden_states[:, 0, :]], dim=1)
            else: 
                output = hidden_states[:, 0, :]

            output = torch.sigmoid(output)
            return output

        output = self.model(x.to("cuda"))
        output = torch.sigmoid(output)
        return output

    def forward(self, input1, input2):

        # Forward
        output1 = self.forward_once(input1.to("cuda"))
        output2 = self.forward_once(input2.to("cuda"))

        # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)

        return output


    def setup_transforms(self, image_size): 
        self.transform_lazy = Compose([
                    Resize((image_size, image_size)),
                    ToTensorIfNotTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def lazy_forward(self, input1, input2, image_size=256): 
        if self.transform_lazy is None: 
            self.setup_transforms(image_size=image_size)

        input1 = self.transform_lazy(input1)
        input2 = self.transform_lazy(input2)

        if input1.dim() == 3: 
            input1 = input1.unsqueeze(dim=0)

        if input2.dim() == 3:
            input2 = input2.unsqueeze(dim=0)

        with torch.no_grad(): 
            out = self.forward(input1, input2)

        return out


    def lazy_forward_once(self, input1, image_size=256): 
        if self.transform_lazy is None: 
            self.setup_transforms(image_size=image_size)

        input1 = self.transform_lazy(input1)
        if input1.dim() == 3: 
            input1 = input1.unsqueeze(dim=0)

        with torch.no_grad(): 
            out = self.forward_once(input1)
        return out


    def feat_to_pred(self, feat1, feat2):
        difference = torch.abs(feat1 - feat2)
        with torch.no_grad():
            output = self.fc_end(difference)
        out = torch.sigmoid(output)
        out = (out > 0.5).int().item()
        return out 


    def lazy_pred(self, input1, input2, image_size=256): 
        output = self.lazy_forward(input1, input2, image_size)
        out = torch.sigmoid(output)
        out = (out > 0.5).int()
        return out


class ImageSpaceSiameseNetwork(SiameseNetwork):
    """
    Siamese Network that computes latents on-the-fly from images.
    Wraps a base SiameseNetwork and adds VAE encoding + sample-wise normalization.
    """
    def __init__(self, network='ConvNeXt-Tiny', in_channels=4, n_features=512, backbone_cfg=None,
                 vae_model_path='stabilityai/stable-diffusion-2', final_mean=0.0, final_std=0.5, distill=False):
        """
        Args:
            network: Base network architecture for SiameseNetwork
            in_channels: Expected input channels (4 for latents)
            n_features: Number of features for the base network
            backbone_cfg: Backbone configuration for base network
            vae_model_path: Path to VAE model for encoding images to latents
            final_mean: Target mean for sample-wise normalization
            final_std: Target std for sample-wise normalization
        """
        # Initialize base SiameseNetwork (expects latents as input)
        super().__init__(network=network, in_channels=in_channels, n_features=n_features, backbone_cfg=backbone_cfg)
        
        # Load VAE model for encoding images to latents
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae").to("cuda").eval()
        self.vae_model_path = vae_model_path
        self.final_mean = final_mean
        self.final_std = final_std
        self.n_features = n_features
        self.distill = distill
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def _apply_sample_wise_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sample-wise normalization to latents.
        Normalizes each sample using its own channel-wise statistics.
        
        Args:
            x: Tensor [B,C,H,W] on GPU
        
        Returns:
            Normalized tensor with mean=final_mean and std=final_std per channel
        """
        # Compute statistics for each sample and channel across spatial dimensions
        # x: [B, C, H, W]
        channel_mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        channel_std = x.std(dim=(2, 3), keepdim=True)    # [B, C, 1, 1]
        
        # Avoid division by zero
        channel_std = torch.clamp(channel_std, min=1e-8)
        
        # Normalize: (x - mean) / std, then scale to target mean/std
        x_normalized = (x - channel_mean) / channel_std
        x_normalized = x_normalized * self.final_std + self.final_mean
        
        return x_normalized
    
    def _encode_to_latent(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using VAE, then apply sample-wise normalization.
        
        Args:
            img: Image tensor [B, C, H, W] in [0, 1] on GPU
            
        Returns:
            Normalized latent tensor [B, 4, H_latent, W_latent] on GPU
        """
        with torch.no_grad():
            # Ensure images are in [0, 1] range
            img = img.clamp(0.0, 1.0)
            # Encode to latent
            latent_dist = self.vae.encode(img)
            latents = latent_dist.latent_dist.sample()  # [B, 4, H_latent, W_latent] on GPU
        
        # Apply sample-wise normalization
        latents = self._apply_sample_wise_normalization(latents)
        return latents
    
    def forward_once_logits(self, x):
        """
        Forward pass: encode image to latent, normalize, then pass through base network.
        """
        # If input is already a latent (matches expected in_channels, typically 4), skip encoding
        # Assume it's already a latent, just normalize
        if x.shape[1] == 3:
            x = self._encode_to_latent(x)
        latents = self._apply_sample_wise_normalization(x)
        output = self.model(latents.to("cuda"))
        # Pass normalized latents through base network
        return output 

    def forward_once(self, x):
        """
        Forward pass: encode image to latent, normalize, then pass through base network.
        """
        # If input is already a latent (matches expected in_channels, typically 4), skip encoding
        # Assume it's already a latent, just normalize
        output = self.forward_once_logits(x)
        output = torch.sigmoid(output)
        # Pass normalized latents through base network
        return output 
    
    def forward(self, input1, input2):
        """
        Forward pass for both inputs: encode images to latents, normalize, then pass through base network.
        """
        if self.distill and input2.size()[1] == self.n_features: 
            # only encode and normalize first input, the other is precomputed feature
            feat1 = self.forward_once(latents1)
            feat2 = input2

            difference = torch.abs(feat1 - feat2)
            output = self.fc_end(difference)
            return output

        # Encode and normalize both inputs
        if input1.shape[1] == 3:
            # Encode images to latents
            input1 = self._encode_to_latent(input1)
            input2 = self._encode_to_latent(input2)

        # Assume already latents, just normalize
        latents1 = self._apply_sample_wise_normalization(input1)
        latents2 = self._apply_sample_wise_normalization(input2)
      
        # Pass through base network
        return super().forward(latents1, latents2) 
