"""
DenseCL Contrastive Head
=========================
Implements InfoNCE contrastive loss with support for:
  - Standard contrastive loss (query vs positive + queue negatives)
  - Per-element weighting (for Soft Lesion-Aware Weighting)
  - Hard negative injection

Reference: https://github.com/WXinlong/DenseCL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """Contrastive head computing InfoNCE loss.
    
    L = -log( exp(q·k+ / τ) / Σ exp(q·k_i / τ) )
    
    With optional per-element weighting for lesion-aware dense loss.
    
    Args:
        temperature (float): Temperature scaling parameter τ.
    """
    
    def __init__(self, temperature=0.2):
        super(ContrastiveHead, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pos, neg, weights=None):
        """Compute InfoNCE contrastive loss.
        
        Args:
            pos (Tensor): Positive logits, shape (N, 1).
            neg (Tensor): Negative logits, shape (N, K) where K = queue_len.
            weights (Tensor, optional): Per-element weights, shape (N,).
                Used for Soft Lesion-Aware Weighting on dense loss.
        
        Returns:
            dict: {'loss_contra': scalar loss}
        """
        N = pos.shape[0]
        
        # Concatenate positive and negative logits
        logits = torch.cat([pos, neg], dim=1)  # (N, 1+K)
        logits /= self.temperature
        
        # Labels: positive is always at index 0
        labels = torch.zeros(N, dtype=torch.long, device=logits.device)
        
        # Compute per-element loss
        loss = self.criterion(logits, labels)  # (N,)
        
        if weights is not None:
            # Apply per-element weighting (lesion-aware)
            assert weights.shape[0] == N, \
                f"Weight shape {weights.shape} doesn't match loss shape {loss.shape}"
            loss = loss * weights
        
        # Mean reduction
        loss = loss.mean()
        
        return {'loss_contra': loss}


class ContrastiveHeadWithHardNeg(nn.Module):
    """Extended contrastive head that supports explicit hard negatives.
    
    In addition to the standard queue negatives, this head can inject
    hard negative features (from normal bone regions) to force the model
    to discriminate IO lesions from similar-looking bone.
    
    Args:
        temperature (float): Temperature scaling parameter τ.
    """
    
    def __init__(self, temperature=0.2):
        super(ContrastiveHeadWithHardNeg, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pos, neg, hard_neg=None, weights=None):
        """Compute InfoNCE with optional hard negatives.
        
        Args:
            pos (Tensor): Positive logits, shape (N, 1).
            neg (Tensor): Negative logits from queue, shape (N, K).
            hard_neg (Tensor, optional): Hard negative logits, shape (N, H) 
                where H = num hard negatives.
            weights (Tensor, optional): Per-element weights, shape (N,).
        
        Returns:
            dict: {'loss_contra': scalar loss}
        """
        N = pos.shape[0]
        
        if hard_neg is not None:
            # Insert hard negatives right after positive
            # This makes them more "visible" in the softmax denominator
            logits = torch.cat([pos, hard_neg, neg], dim=1)  # (N, 1+H+K)
        else:
            logits = torch.cat([pos, neg], dim=1)  # (N, 1+K)
        
        logits /= self.temperature
        
        # Labels: positive is always at index 0
        labels = torch.zeros(N, dtype=torch.long, device=logits.device)
        
        # Per-element loss
        loss = self.criterion(logits, labels)  # (N,)
        
        if weights is not None:
            loss = loss * weights
        
        loss = loss.mean()
        
        return {'loss_contra': loss}
