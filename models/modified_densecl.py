"""
Modified DenseCL for Idiopathic Osteosclerosis Detection
=========================================================
Core module implementing DenseCL with three modifications:
  1. Soft Lesion-Aware Weighting — dense loss weighted higher on lesion regions
  2. Asymmetric Augmentation — handled in dataset/transforms (view1=global, view2=local)
  3. Radiologically-Informed Hard Negative Mining — bone-normal regions as hard negatives

Architecture follows MoCo-style momentum encoder with dual queues
(one for global, one for dense features).

Reference: Wang et al., "Dense Contrastive Learning for Self-Supervised 
           Visual Pre-Training", CVPR 2021.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNetBackbone
from .densecl_neck import DenseCLNeck
from .densecl_head import ContrastiveHead, ContrastiveHeadWithHardNeg


class ModifiedDenseCL(nn.Module):
    """Modified DenseCL with lesion-aware weighting and hard negative mining.
    
    This model performs self-supervised contrastive learning at both global
    and dense (pixel-level) granularities. Three modifications are introduced
    to improve representation learning for small IO lesions in dental panoramic images.
    
    Args:
        backbone_cfg (dict): Backbone configuration.
        neck_cfg (dict): Neck configuration. 
        head_cfg (dict): Head configuration.
        queue_len (int): Size of the negative sample queue.
        feat_dim (int): Feature embedding dimension.
        momentum (float): Momentum coefficient for EMA encoder update.
        loss_lambda (float): Weight between global and dense loss. 
            total_loss = (1-λ)*L_global + λ*L_dense
        
        # --- Modification 1: Lesion-Aware Weighting ---
        use_lesion_weighting (bool): Enable soft lesion-aware weighting.
        lesion_alpha (float): Weight multiplier for lesion regions.
        lesion_gaussian_sigma (float): Gaussian smoothing sigma for boundary.
        
        # --- Modification 3: Hard Negative Mining ---
        use_hard_negative_mining (bool): Enable hard negative mining.
        hard_neg_warmup_epochs (int): Warmup epochs before activation.
        hard_neg_top_k (int): Number of hard negatives per query.
    """
    
    def __init__(self,
                 backbone_cfg=None,
                 neck_cfg=None,
                 head_cfg=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda=0.5,
                 # Modification 1
                 use_lesion_weighting=True,
                 lesion_alpha=2.0,
                 lesion_gaussian_sigma=3.0,
                 # Modification 3
                 use_hard_negative_mining=True,
                 hard_neg_warmup_epochs=20,
                 hard_neg_top_k=5,
                 hard_neg_loss_weight=0.1):
        super(ModifiedDenseCL, self).__init__()
        
        # Default configs
        if backbone_cfg is None:
            backbone_cfg = dict(depth=50, pretrained=False, return_stages=[4])
        if neck_cfg is None:
            neck_cfg = dict(in_channels=2048, hid_channels=2048, 
                          out_channels=feat_dim, num_grid=None)
        if head_cfg is None:
            head_cfg = dict(temperature=0.2)
        
        # =====================================================================
        # Build encoder_q (query) and encoder_k (key, momentum-updated)
        # =====================================================================
        self.backbone_q = ResNetBackbone(**backbone_cfg)
        self.neck_q = DenseCLNeck(**neck_cfg)
        
        # Key encoder is a copy — no gradients
        self.backbone_k = copy.deepcopy(self.backbone_q)
        self.neck_k = copy.deepcopy(self.neck_q)
        
        for param in self.backbone_k.parameters():
            param.requires_grad = False
        for param in self.neck_k.parameters():
            param.requires_grad = False
        
        # Keep a reference for backbone extraction
        self.backbone = self.backbone_q
        
        # =====================================================================
        # Contrastive heads
        # =====================================================================
        if use_hard_negative_mining:
            self.head = ContrastiveHeadWithHardNeg(**head_cfg)
        else:
            self.head = ContrastiveHead(**head_cfg)
        
        # =====================================================================
        # Queues (MoCo-style)
        # =====================================================================
        self.queue_len = queue_len
        self.momentum = momentum
        self.loss_lambda = loss_lambda
        self.feat_dim = feat_dim
        
        # Queue 1: Global features
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Queue 2: Dense features (average pooled)
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))
        
        # =====================================================================
        # Modification parameters
        # =====================================================================
        self.use_lesion_weighting = use_lesion_weighting
        self.lesion_alpha = lesion_alpha
        self.lesion_gaussian_sigma = lesion_gaussian_sigma
        
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_neg_warmup_epochs = hard_neg_warmup_epochs
        self.hard_neg_top_k = hard_neg_top_k
        self.hard_neg_loss_weight = hard_neg_loss_weight
        
        # Current epoch (updated externally during training)
        self.current_epoch = 0
        
        # Hard negative feature bank
        if use_hard_negative_mining:
            self.register_buffer(
                "hard_neg_bank", 
                torch.randn(feat_dim, 1000)  # Will be resized dynamically
            )
            self.hard_neg_bank = F.normalize(self.hard_neg_bank, dim=0)
            self.register_buffer(
                "hard_neg_bank_ptr", 
                torch.zeros(1, dtype=torch.long)
            )
            self.hard_neg_bank_size = 1000
        
        # Initialize neck weights
        self.neck_q.init_weights()
        self._init_key_encoder()
    
    def _init_key_encoder(self):
        """Initialize key encoder with query encoder weights."""
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                     self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.neck_q.parameters(),
                                     self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update: θ_k = m * θ_k + (1 - m) * θ_q"""
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                     self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.neck_q.parameters(),
                                     self.neck_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        """Replace oldest entries in queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        
        # Handle case where queue_len is not divisible by batch_size
        if ptr + batch_size > self.queue_len:
            remaining = self.queue_len - ptr
            queue[:, ptr:] = keys[:remaining].T
            overflow = batch_size - remaining
            queue[:, :overflow] = keys[remaining:].T
            ptr = overflow
        else:
            queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_len
        
        queue_ptr[0] = ptr
    
    def _compute_lesion_weights(self, weight_maps, spatial_size):
        """Compute per-position weights for dense loss using lesion weight maps.
        
        Args:
            weight_maps (Tensor or None): Weight maps from dataset, shape (B, H_orig, W_orig).
                Values > 1.0 indicate lesion regions (after Gaussian smoothing).
            spatial_size (int): Number of spatial positions (S^2) in dense features.
        
        Returns:
            Tensor: Flattened weights (B*S^2,) for dense loss.
        """
        if weight_maps is None or not self.use_lesion_weighting:
            return None
        
        B = weight_maps.shape[0]
        
        # Determine spatial grid dimensions
        # For landscape images, H and W of feature maps differ
        # We need to figure out the spatial dimensions from the backbone output
        # Since we flatten to S^2, we can just interpolate the weight map
        # to the total number of spatial positions
        S2 = spatial_size
        
        # Interpolate weight maps to match spatial grid
        # weight_maps: (B, H_orig, W_orig) -> need to match feature map spatial
        if weight_maps.dim() == 2:
            # If 1D, it's already per-position
            weights = weight_maps
        else:
            # Resize to approximate spatial dims
            # For ResNet-50 with input (256, 512), layer4 output is (8, 16) = 128 positions
            H_feat = int((S2 ** 0.5) * 0.5)  # rough estimate for landscape
            W_feat = S2 // max(H_feat, 1)
            if H_feat * W_feat != S2:
                # Fallback: treat as square
                H_feat = int(S2 ** 0.5)
                W_feat = S2 // max(H_feat, 1)
                if H_feat * W_feat != S2:
                    H_feat = 1
                    W_feat = S2
            
            weight_maps_4d = weight_maps.unsqueeze(1).float()  # (B, 1, H, W)
            resized = F.interpolate(weight_maps_4d, size=(H_feat, W_feat),
                                   mode='bilinear', align_corners=False)
            weights = resized.view(B, -1)  # (B, S^2)
        
        # Flatten for dense loss (B*S^2,)
        weights = weights.reshape(-1)
        
        return weights
    
    def _mine_hard_negatives(self, q_grid, q_b):
        """Mine hard negatives from the hard negative feature bank.
        
        Hard negatives are bone-normal regions that look similar to IO lesions
        both in pixel intensity and feature space.
        
        Args:
            q_grid (Tensor): Query dense features (B, C, S^2), normalized.
            q_b (Tensor): Query backbone features (B, C_backbone, S^2), normalized.
        
        Returns:
            Tensor or None: Hard negative logits (B*S^2, top_k) or None if not active.
        """
        if not self.use_hard_negative_mining:
            return None
        
        if self.current_epoch < self.hard_neg_warmup_epochs:
            return None
        
        B, C, S2 = q_grid.shape
        
        # Compute similarity between each query position and hard neg bank
        q_flat = q_grid.permute(0, 2, 1).reshape(-1, C)  # (B*S^2, C)
        
        # Cosine similarity with hard neg bank
        sim = torch.mm(q_flat, self.hard_neg_bank[:C, :])  # (B*S^2, bank_size)
        
        # Select top-K most similar (hardest) negatives
        top_k = min(self.hard_neg_top_k, sim.shape[1])
        _, hard_neg_indices = sim.topk(top_k, dim=1)  # (B*S^2, top_k)
        
        # Gather hard negative features
        hard_neg_feats = self.hard_neg_bank[:C, :].T  # (bank_size, C)
        hard_neg_selected = hard_neg_feats[hard_neg_indices.view(-1)]  # (B*S^2*top_k, C)
        hard_neg_selected = hard_neg_selected.view(B * S2, top_k, C)
        
        # Compute logits: (B*S^2, top_k)
        hard_neg_logits = torch.bmm(
            q_flat.unsqueeze(1),  # (B*S^2, 1, C)
            hard_neg_selected.permute(0, 2, 1)  # (B*S^2, C, top_k)
        ).squeeze(1)  # (B*S^2, top_k)
        
        return hard_neg_logits
    
    @torch.no_grad()
    def update_hard_neg_bank(self, features):
        """Update the hard negative feature bank with new bone-normal features.
        
        Called externally during training to populate the bank with features
        from regions identified as normal bone (not lesion-like).
        
        Args:
            features (Tensor): Normalized feature vectors (N, C).
        """
        if not self.use_hard_negative_mining:
            return
        
        N = features.shape[0]
        if N == 0:
            return
        
        C = features.shape[1]
        
        # Resize bank if needed
        if self.hard_neg_bank.shape[0] < C:
            device = self.hard_neg_bank.device
            self.hard_neg_bank = torch.randn(C, self.hard_neg_bank_size, device=device)
            self.hard_neg_bank = F.normalize(self.hard_neg_bank, dim=0)
        
        ptr = int(self.hard_neg_bank_ptr)
        
        if ptr + N > self.hard_neg_bank_size:
            remaining = self.hard_neg_bank_size - ptr
            self.hard_neg_bank[:C, ptr:] = features[:remaining].T
            overflow = min(N - remaining, self.hard_neg_bank_size)
            self.hard_neg_bank[:C, :overflow] = features[remaining:remaining+overflow].T
            self.hard_neg_bank_ptr[0] = overflow
        else:
            self.hard_neg_bank[:C, ptr:ptr + N] = features.T
            self.hard_neg_bank_ptr[0] = (ptr + N) % self.hard_neg_bank_size
    
    def forward_train(self, im_q, im_k, weight_maps=None):
        """Forward pass for training.
        
        Args:
            im_q (Tensor): Query images (View 1 — global crop), shape (B, 3, H, W).
            im_k (Tensor): Key images (View 2 — local or global crop), shape (B, 3, H', W').
            weight_maps (Tensor, optional): Lesion weight maps for View 1, 
                shape (B, H_map, W_map). Values > 1.0 for lesion regions.
        
        Returns:
            dict: Loss dictionary with 'loss_contra_single' and 'loss_contra_dense'.
        """
        # =================================================================
        # Encode query (with gradients)
        # =================================================================
        q_backbone_feats = self.backbone_q(im_q)  # list of [feat]
        q, q_grid, q2 = self.neck_q(q_backbone_feats)  # global, dense_grid, dense_avg
        
        # Also get raw backbone features for dense matching
        q_b = q_backbone_feats[0]  # (B, 2048, H_feat, W_feat)
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)  # (B, 2048, S^2)
        
        # Normalize all embeddings
        q = F.normalize(q, dim=1)        # (B, C)
        q2 = F.normalize(q2, dim=1)      # (B, C)
        q_grid = F.normalize(q_grid, dim=1)  # (B, C, S^2)
        q_b = F.normalize(q_b, dim=1)    # (B, 2048, S^2)
        
        # =================================================================
        # Encode key (no gradients, momentum-updated)
        # =================================================================
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k_backbone_feats = self.backbone_k(im_k)
            k, k_grid, k2 = self.neck_k(k_backbone_feats)
            
            k_b = k_backbone_feats[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)
            
            k = F.normalize(k, dim=1)
            k2 = F.normalize(k2, dim=1)
            k_grid = F.normalize(k_grid, dim=1)
            k_b = F.normalize(k_b, dim=1)
        
        # =================================================================
        # Global contrastive loss (same as MoCo)
        # =================================================================
        # Positive logits: (B, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: (B, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        loss_single = self.head(l_pos, l_neg)['loss_contra']
        
        # =================================================================
        # Dense contrastive loss (DenseCL core)
        # =================================================================
        B = q_b.shape[0]
        S2_q = q_b.shape[2]  # spatial positions in query
        S2_k = k_b.shape[2]  # spatial positions in key (may differ for asymmetric)
        
        # Find best-matching key position for each query position
        # backbone_sim_matrix: (B, S^2_q, S^2_k)
        backbone_sim_matrix = torch.bmm(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # (B, S^2_q)
        
        # Gather matched key grid features
        # indexed_k_grid: (B, C, S^2_q) — for each query position, the best-matching key feature
        indexed_k_grid = torch.gather(
            k_grid, 2, 
            densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)
        )
        
        # Dense positive logits
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # (B, S^2_q)
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)  # (B*S^2, 1)
        
        # Dense negative logits (using queue2)
        q_grid_flat = q_grid.permute(0, 2, 1).reshape(-1, q_grid.size(1))  # (B*S^2, C)
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid_flat, 
                                                   self.queue2.clone().detach()])
        
        # =================================================================
        # Modification 1: Soft Lesion-Aware Weighting
        # =================================================================
        dense_weights = self._compute_lesion_weights(weight_maps, S2_q)
        
        # =================================================================
        # Modification 3: Hard Negative Mining
        # =================================================================
        hard_neg_logits = self._mine_hard_negatives(q_grid, q_b)
        
        # Compute dense loss
        if isinstance(self.head, ContrastiveHeadWithHardNeg):
            loss_dense = self.head(
                l_pos_dense, l_neg_dense, 
                hard_neg=hard_neg_logits,
                weights=dense_weights
            )['loss_contra']
        else:
            loss_dense = self.head(
                l_pos_dense, l_neg_dense, 
                weights=dense_weights
            )['loss_contra']
        
        # =================================================================
        # Combine losses
        # =================================================================
        losses = {}
        losses['loss_contra_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda
        losses['loss_total'] = losses['loss_contra_single'] + losses['loss_contra_dense']
        
        # =================================================================
        # Update queues
        # =================================================================
        self._dequeue_and_enqueue(k, self.queue, self.queue_ptr)
        self._dequeue_and_enqueue(k2, self.queue2, self.queue2_ptr)
        
        # =================================================================
        # Update hard negative bank (collect normal bone features)
        # =================================================================
        if self.use_hard_negative_mining and weight_maps is not None:
            with torch.no_grad():
                # Extract features from non-lesion regions as hard neg candidates
                # Use positions where weight_map ≈ 1.0 (background/normal bone)
                flat_weights = self._compute_lesion_weights(weight_maps, S2_q)
                if flat_weights is not None:
                    # Normal bone = weight close to 1.0 (not boosted)
                    normal_mask = flat_weights < 1.1  # background weight
                    if normal_mask.any():
                        normal_feats = q_grid_flat[normal_mask].detach()
                        normal_feats = F.normalize(normal_feats, dim=1)
                        # Keep only most "dense-looking" normal regions
                        # (high similarity to lesion-like patterns)
                        if normal_feats.shape[0] > 100:
                            # Sample a subset
                            indices = torch.randperm(normal_feats.shape[0])[:100]
                            normal_feats = normal_feats[indices]
                        self.update_hard_neg_bank(normal_feats)
        
        return losses
    
    def forward_test(self, img):
        """Forward pass for testing / feature extraction.
        
        Args:
            img (Tensor): Input images (B, 3, H, W).
        
        Returns:
            Tensor: Dense feature grid, normalized. (B, C, S^2)
        """
        backbone_feats = self.backbone_q(img)
        q, q_grid, q2 = self.neck_q(backbone_feats)
        q_grid = F.normalize(q_grid, dim=1)
        return q_grid
    
    def forward(self, im_q, im_k=None, mode='train', weight_maps=None):
        """Main forward dispatch.
        
        Args:
            im_q: Query image (View 1).
            im_k: Key image (View 2), only needed for training.
            mode: 'train', 'test', or 'extract'.
            weight_maps: Lesion weight maps (only for training with Mod 1).
        """
        if mode == 'train':
            assert im_k is not None, "im_k required for training"
            return self.forward_train(im_q, im_k, weight_maps)
        elif mode == 'test':
            return self.forward_test(im_q)
        elif mode == 'extract':
            return self.backbone_q(im_q)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def extract_backbone_weights(self):
        """Extract backbone state dict for downstream use.
        
        Returns:
            dict: State dict of the query backbone.
        """
        return self.backbone_q.state_dict()
