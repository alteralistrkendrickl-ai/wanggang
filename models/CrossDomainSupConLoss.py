"""Cross-domain supervised contrastive consistency for emitter features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDomainSupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def forward(self, features, identity_labels, domain_labels):
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, feature_dim)")
        identity_labels = identity_labels.reshape(-1)
        domain_labels = domain_labels.reshape(-1)
        if not (len(features) == len(identity_labels) == len(domain_labels)):
            raise ValueError("features, identity labels, and domain labels must align")
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature
        identity_equal = identity_labels[:, None].eq(identity_labels[None, :])
        domain_different = domain_labels[:, None].ne(domain_labels[None, :])
        eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
        positive_mask = identity_equal & domain_different
        denominator_mask = ((~identity_equal) | positive_mask) & (~eye)
        positive_count = positive_mask.sum(dim=1)
        if torch.any(positive_count == 0):
            bad = torch.nonzero(positive_count == 0).reshape(-1).tolist()
            raise ValueError(f"A1 batch contains anchors without cross-domain positives: {bad}")
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * denominator_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_positive_log_prob = (log_prob * positive_mask).sum(dim=1) / positive_count
        return -mean_positive_log_prob.mean()


def create_model(temperature=0.1):
    return CrossDomainSupConLoss(temperature=temperature)
