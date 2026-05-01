"""
ResNet-50 Backbone for DenseCL
===============================
Wraps torchvision ResNet-50, returning intermediate feature maps
suitable for both DenseCL pretraining and Faster R-CNN downstream.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone that returns feature maps from specified stages.
    
    For DenseCL pretraining:
        - Returns only layer4 output (2048 channels, H/32 × W/32 spatial)
    
    For Faster R-CNN (FPN):
        - Returns feature maps from layer1-4 for FPN
    
    Args:
        depth (int): ResNet depth, currently only 50 is supported.
        pretrained (bool): Whether to use ImageNet pretrained weights.
        return_stages (list): Which stages to return (1-4). 
            Default [4] for DenseCL, [1,2,3,4] for FPN.
        frozen_stages (int): Freeze stages up to this number. -1 means no freezing.
    """
    
    def __init__(self, depth=50, pretrained=False, return_stages=None, 
                 frozen_stages=-1):
        super(ResNetBackbone, self).__init__()
        
        if return_stages is None:
            return_stages = [4]
        
        assert depth == 50, f"Only ResNet-50 is supported, got depth={depth}"
        
        # Build the base ResNet
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        self.return_stages = return_stages
        self.frozen_stages = frozen_stages
        
        # Decompose ResNet into stages
        self.conv1 = resnet.conv1       # 3 -> 64, stride 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool    # stride 2
        
        self.layer1 = resnet.layer1     # 64 -> 256,  H/4
        self.layer2 = resnet.layer2     # 256 -> 512,  H/8
        self.layer3 = resnet.layer3     # 512 -> 1024, H/16
        self.layer4 = resnet.layer4     # 1024 -> 2048, H/32
        
        # Stage-to-layer mapping
        self._stages = {
            1: self.layer1,
            2: self.layer2,
            3: self.layer3,
            4: self.layer4,
        }
        
        # Output channels per stage
        self.stage_channels = {
            1: 256,
            2: 512,
            3: 1024,
            4: 2048,
        }
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze parameters up to frozen_stages."""
        if self.frozen_stages >= 0:
            # Freeze stem (conv1, bn1)
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
        
        for i in range(1, self.frozen_stages + 1):
            layer = self._stages[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
    
    def init_weights(self, pretrained=None):
        """Initialize weights, optionally loading a pretrained checkpoint."""
        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Filter and load matching keys
            model_dict = self.state_dict()
            filtered = {k: v for k, v in state_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(filtered)}/{len(model_dict)} parameters from {pretrained}")
    
    def forward(self, x):
        """Forward pass returning feature maps from specified stages.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            list of feature maps from return_stages
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outputs = []
        
        # Stage 1-4
        x = self.layer1(x)
        if 1 in self.return_stages:
            outputs.append(x)
        
        x = self.layer2(x)
        if 2 in self.return_stages:
            outputs.append(x)
        
        x = self.layer3(x)
        if 3 in self.return_stages:
            outputs.append(x)
        
        x = self.layer4(x)
        if 4 in self.return_stages:
            outputs.append(x)
        
        return outputs
    
    def train(self, mode=True):
        """Override train to keep frozen stages in eval mode."""
        super(ResNetBackbone, self).train(mode)
        self._freeze_stages()
        return self
