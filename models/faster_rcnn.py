"""
Faster R-CNN Wrapper for Downstream IO Detection
==================================================
Uses torchvision's Faster R-CNN with FPN, loading the pretrained
ResNet-50 backbone from Modified DenseCL pretraining.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops

from .resnet import ResNetBackbone


def build_faster_rcnn(num_classes=2, 
                      pretrained_backbone_path=None,
                      min_size=512, 
                      max_size=1024,
                      trainable_backbone_layers=3):
    """Build Faster R-CNN with FPN using pretrained DenseCL backbone.
    
    Args:
        num_classes (int): Number of classes (including background).
            For IO detection: 2 (background + IO).
        pretrained_backbone_path (str): Path to extracted backbone weights 
            from DenseCL pretraining. If None, uses random init.
        min_size (int): Minimum size of the image to be rescaled.
        max_size (int): Maximum size of the image to be rescaled.
        trainable_backbone_layers (int): Number of trainable (not frozen) 
            ResNet layers starting from layer4. Default 3 means 
            layer2, layer3, layer4 are trainable.
    
    Returns:
        FasterRCNN: Model ready for fine-tuning.
    """
    # Build ResNet-50 backbone with FPN outputs
    backbone = ResNetBackbone(
        depth=50, pretrained=False, 
        return_stages=[1, 2, 3, 4],
        frozen_stages=-1
    )
    
    # Load pretrained weights from DenseCL
    if pretrained_backbone_path is not None:
        state_dict = torch.load(pretrained_backbone_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter matching keys
        model_dict = backbone.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            # Handle prefix differences
            clean_key = k.replace('backbone_q.', '').replace('backbone.', '')
            if clean_key in model_dict and v.shape == model_dict[clean_key].shape:
                filtered[clean_key] = v
        
        model_dict.update(filtered)
        backbone.load_state_dict(model_dict, strict=False)
        print(f"[Faster R-CNN] Loaded {len(filtered)}/{len(model_dict)} "
              f"backbone params from {pretrained_backbone_path}")
    
    # Freeze early layers based on trainable_backbone_layers
    layers_to_freeze = 4 - trainable_backbone_layers
    if layers_to_freeze > 0:
        backbone.frozen_stages = layers_to_freeze
        backbone._freeze_stages()
    
    # Wrap backbone with FPN
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = [256, 512, 1024, 2048]
    
    backbone_with_fpn = _BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels=256
    )
    
    # Anchor generator for dental panoramic images
    # IO lesions are typically small, so use smaller anchor sizes
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios
    )
    
    # ROI pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Build Faster R-CNN
    model = FasterRCNN(
        backbone_with_fpn,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # RPN settings
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        # Detection settings
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
    )
    
    return model


class _BackboneWithFPN(nn.Module):
    """Custom BackboneWithFPN wrapper for our ResNetBackbone.
    
    torchvision's BackboneWithFPN expects IntermediateLayerGetter,
    but our backbone already returns specified stages. This wrapper
    bridges the gap.
    """
    
    def __init__(self, backbone, return_layers, in_channels_list, 
                 out_channels=256):
        super(_BackboneWithFPN, self).__init__()
        
        self.body = backbone
        self.out_channels = out_channels
        
        # FPN layers
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list, out_channels,
            extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
        )
    
    def forward(self, x):
        """Forward pass through backbone + FPN.
        
        Args:
            x (Tensor): Input image (B, 3, H, W).
        
        Returns:
            OrderedDict: FPN feature maps {'0': P2, '1': P3, '2': P4, '3': P5, 'pool': P6}
        """
        from collections import OrderedDict
        
        # Get multi-scale features from backbone
        features = self.body(x)  # list of 4 feature maps
        
        # Convert to ordered dict for FPN
        feat_dict = OrderedDict()
        for i, feat in enumerate(features):
            feat_dict[str(i)] = feat
        
        # Apply FPN
        fpn_features = self.fpn(feat_dict)
        
        return fpn_features
