"""
DenseCL Projection Neck
========================
Implements the dual-branch projection neck from DenseCL:
  - Global branch: AdaptiveAvgPool -> MLP (fc-relu-fc)
  - Dense branch: Conv1x1-relu-Conv1x1 (preserves spatial structure)

Reference: https://github.com/WXinlong/DenseCL
"""

import torch
import torch.nn as nn


class DenseCLNeck(nn.Module):
    """DenseCL projection neck with single (global) and dense (spatial) branches.
    
    The neck takes backbone features (B, C, H, W) and produces:
        1. global_embedding (B, out_channels) — pooled global representation
        2. dense_grid (B, out_channels, S^2) — spatial grid of dense embeddings
        3. dense_avg (B, out_channels) — average of dense embeddings
    
    Args:
        in_channels (int): Number of input channels from backbone (2048 for ResNet-50 layer4).
        hid_channels (int): Hidden dimension in projector MLPs.
        out_channels (int): Output embedding dimension (feat_dim).
        num_grid (int or None): If set, pool dense features to (num_grid, num_grid).
            If None, use the backbone's native spatial dimensions.
    """
    
    def __init__(self, in_channels=2048, hid_channels=2048, out_channels=128,
                 num_grid=None):
        super(DenseCLNeck, self).__init__()
        
        # --- Global branch: pool + MLP ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels)
        )
        
        # --- Dense branch: Conv1x1 MLP (preserves spatial) ---
        self.with_pool = num_grid is not None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, kernel_size=1)
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
    
    def init_weights(self, init_linear='kaiming'):
        """Initialize weights using kaiming or normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', 
                                           nonlinearity='relu')
                else:
                    nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (list): List containing single feature map [feat] where 
                      feat is (B, C, H, W) from backbone layer4.
        
        Returns:
            list: [global_embedding, dense_grid, dense_avg]
                - global_embedding: (B, out_channels)
                - dense_grid: (B, out_channels, S^2) where S^2 = H*W
                - dense_avg: (B, out_channels)
        """
        assert len(x) == 1, f"Expected 1 feature map, got {len(x)}"
        x = x[0]  # (B, C, H, W)
        
        # --- Global branch ---
        avgpooled_x = self.avgpool(x)  # (B, C, 1, 1)
        global_embedding = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))  # (B, out_channels)
        
        # --- Dense branch ---
        if self.with_pool:
            x = self.pool(x)  # (B, C, num_grid, num_grid)
        
        dense_feat = self.mlp2(x)  # (B, out_channels, H', W')
        dense_avg = self.avgpool2(dense_feat)  # (B, out_channels, 1, 1)
        
        # Flatten spatial dims
        dense_grid = dense_feat.view(dense_feat.size(0), dense_feat.size(1), -1)  # (B, out_channels, S^2)
        dense_avg = dense_avg.view(dense_avg.size(0), -1)  # (B, out_channels)
        
        return [global_embedding, dense_grid, dense_avg]
