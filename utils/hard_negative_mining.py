"""
Radiologically-Informed Hard Negative Mining (Modification 3)
===============================================================
Mines hard negatives from normal bone regions that visually and
semantically resemble IO lesions. Forces the model to learn subtle
differences between true IO and similar-looking normal bone.

Key design:
  - Warmup period (20 epochs) lets the model learn basic representations first
  - Combined scoring: pixel intensity similarity + cosine similarity in feature space
  - Feature bank stores normal bone features, updated periodically
  - Hard negatives are injected into the dense contrastive loss denominator
"""

import torch
import torch.nn.functional as F
import numpy as np


class HardNegativeMiner:
    """Mines and manages hard negative samples for contrastive learning.
    
    Hard negatives are normal bone regions whose features are highly similar
    to IO lesion features, making them challenging negative examples that
    push the model to learn more discriminative representations.
    
    The miner maintains a feature bank of normal bone region features
    and identifies the most confusing (similar) ones as hard negatives.
    
    Args:
        feat_dim (int): Feature embedding dimension.
        bank_size (int): Maximum number of features in the bank.
        top_k (int): Number of hard negatives to select per query.
        warmup_epochs (int): Number of warmup epochs before activation.
        intensity_weight (float): Weight for pixel intensity similarity.
        feature_weight (float): Weight for feature space cosine similarity.
        device (str): Device for tensors.
    """
    
    def __init__(self, feat_dim=128, bank_size=10000, top_k=5,
                 warmup_epochs=20, intensity_weight=0.3, 
                 feature_weight=0.7, device='cuda'):
        self.feat_dim = feat_dim
        self.bank_size = bank_size
        self.top_k = top_k
        self.warmup_epochs = warmup_epochs
        self.intensity_weight = intensity_weight
        self.feature_weight = feature_weight
        self.device = device
        
        # Feature bank for normal bone regions
        self.feature_bank = torch.randn(bank_size, feat_dim).to(device)
        self.feature_bank = F.normalize(self.feature_bank, dim=1)
        self.bank_ptr = 0
        self.bank_filled = 0  # Track how many real features are stored
        
        # Intensity statistics bank (mean intensity per region)
        self.intensity_bank = torch.zeros(bank_size).to(device)
        
        # Current epoch
        self.current_epoch = 0
    
    @property
    def is_active(self):
        """Whether hard negative mining is active (past warmup)."""
        return self.current_epoch >= self.warmup_epochs
    
    def set_epoch(self, epoch):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = epoch
    
    @torch.no_grad()
    def update_bank(self, features, intensities=None):
        """Add new normal bone features to the bank.
        
        Called during training to populate the bank with features
        extracted from non-lesion (normal bone) regions.
        
        Args:
            features (Tensor): Normalized feature vectors (N, feat_dim).
            intensities (Tensor, optional): Mean pixel intensities (N,) 
                of the corresponding regions.
        """
        N = features.shape[0]
        if N == 0:
            return
        
        features = features.detach().to(self.device)
        features = F.normalize(features, dim=1)
        
        if intensities is not None:
            intensities = intensities.detach().to(self.device)
        else:
            intensities = torch.zeros(N, device=self.device)
        
        # Insert into bank (circular buffer)
        for i in range(N):
            self.feature_bank[self.bank_ptr] = features[i]
            self.intensity_bank[self.bank_ptr] = intensities[i] if intensities is not None else 0
            self.bank_ptr = (self.bank_ptr + 1) % self.bank_size
            self.bank_filled = min(self.bank_filled + 1, self.bank_size)
    
    @torch.no_grad()
    def mine(self, query_features, query_intensities=None):
        """Mine hard negatives for given query features.
        
        Args:
            query_features (Tensor): Query feature vectors (N, feat_dim), normalized.
            query_intensities (Tensor, optional): Mean pixel intensities (N,)
                of query regions.
        
        Returns:
            Tensor or None: Hard negative features (N, top_k, feat_dim).
                Returns None if miner is not active or bank is empty.
        """
        if not self.is_active or self.bank_filled == 0:
            return None
        
        N = query_features.shape[0]
        effective_bank_size = min(self.bank_filled, self.bank_size)
        
        query_features = query_features.to(self.device)
        query_features = F.normalize(query_features, dim=1)
        
        # ===== Feature-space cosine similarity =====
        # (N, feat_dim) × (feat_dim, bank_size) -> (N, bank_size)
        bank_feats = self.feature_bank[:effective_bank_size]
        feature_sim = torch.mm(query_features, bank_feats.T)
        
        # ===== Pixel intensity similarity =====
        if query_intensities is not None:
            query_int = query_intensities.to(self.device).unsqueeze(1)  # (N, 1)
            bank_int = self.intensity_bank[:effective_bank_size].unsqueeze(0)  # (1, bank)
            # Normalized inverse distance (higher = more similar intensity)
            int_diff = torch.abs(query_int - bank_int)
            int_sim = 1.0 / (1.0 + int_diff / 50.0)  # Scale by typical intensity range
        else:
            int_sim = torch.zeros(N, effective_bank_size, device=self.device)
        
        # ===== Combined scoring =====
        combined_score = (self.feature_weight * feature_sim + 
                         self.intensity_weight * int_sim)
        
        # Select top-K most similar (hardest negatives)
        k = min(self.top_k, effective_bank_size)
        _, top_indices = combined_score.topk(k, dim=1)  # (N, k)
        
        # Gather hard negative features
        # top_indices: (N, k) -> gather from bank_feats (bank_size, feat_dim)
        hard_negs = bank_feats[top_indices.view(-1)]  # (N*k, feat_dim)
        hard_negs = hard_negs.view(N, k, self.feat_dim)  # (N, k, feat_dim)
        
        return hard_negs
    
    @torch.no_grad()
    def compute_hard_neg_logits(self, query_features, query_intensities=None):
        """Compute logits between queries and their hard negatives.
        
        Args:
            query_features (Tensor): Query features (N, feat_dim), normalized.
            query_intensities (Tensor, optional): Query intensities (N,).
        
        Returns:
            Tensor or None: Hard negative logits (N, top_k).
        """
        hard_negs = self.mine(query_features, query_intensities)
        if hard_negs is None:
            return None
        
        N, K, C = hard_negs.shape
        
        # Compute logits: (N, 1, C) × (N, C, K) -> (N, 1, K) -> (N, K)
        logits = torch.bmm(
            query_features.unsqueeze(1),  # (N, 1, C)
            hard_negs.permute(0, 2, 1)    # (N, C, K)
        ).squeeze(1)  # (N, K)
        
        return logits
    
    @torch.no_grad()
    def extract_normal_bone_features(self, backbone_features, weight_maps, 
                                      dense_features):
        """Extract features from normal bone regions for bank update.
        
        Identifies spatial positions in the feature map that correspond
        to normal bone (weight ≈ 1.0) and extracts their dense features.
        
        Args:
            backbone_features (Tensor): Raw backbone features (B, C_bb, H, W).
            weight_maps (Tensor): Lesion weight maps (B, H_w, W_w).
            dense_features (Tensor): Dense projection features (B, C, S^2).
        
        Returns:
            tuple: (normal_features, normal_intensities)
                - normal_features: (M, C) normalized features
                - normal_intensities: (M,) mean intensities
        """
        B, C, S2 = dense_features.shape
        
        # Compute spatial dims
        H_bb, W_bb = backbone_features.shape[2], backbone_features.shape[3]
        
        # Resize weight maps to match feature spatial dims
        wm = weight_maps.unsqueeze(1).float()  # (B, 1, H_w, W_w)
        wm_resized = F.interpolate(wm, size=(H_bb, W_bb), mode='bilinear',
                                    align_corners=False)
        wm_flat = wm_resized.squeeze(1).view(B, -1)  # (B, S^2)
        
        # Normal bone = positions where weight is approximately 1.0
        normal_mask = (wm_flat < 1.1) & (wm_flat > 0.9)
        
        # Extract dense features at normal positions
        dense_flat = dense_features.permute(0, 2, 1).reshape(-1, C)  # (B*S^2, C)
        mask_flat = normal_mask.reshape(-1)  # (B*S^2,)
        
        normal_feats = dense_flat[mask_flat]  # (M, C)
        normal_feats = F.normalize(normal_feats, dim=1)
        
        # Compute mean intensity at normal positions from backbone features
        bb_flat = backbone_features.view(B, -1, H_bb * W_bb)  # (B, C_bb, S^2)
        bb_mean = bb_flat.mean(dim=1).reshape(-1)  # (B*S^2,) mean over channels
        normal_intensities = bb_mean[mask_flat]
        
        # Subsample if too many
        max_samples = 500
        if normal_feats.shape[0] > max_samples:
            indices = torch.randperm(normal_feats.shape[0])[:max_samples]
            normal_feats = normal_feats[indices]
            normal_intensities = normal_intensities[indices]
        
        return normal_feats, normal_intensities
