import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class LightweightLFDB(nn.Module):
    """Split encoder output into device-fingerprint and nuisance features."""

    def __init__(self, feat_dim=1024, hidden_dim=128, num_classes=90, grl_alpha=1.0):
        super().__init__()
        self.grl_alpha = grl_alpha
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Linear(hidden_dim, feat_dim)
        self.adv_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        shared = self.shared(features)
        mask = torch.sigmoid(self.mask_head(shared))
        fingerprint_features = features * mask
        nuisance_features = features * (1.0 - mask)
        adversarial_features = GradientReversal.apply(nuisance_features, self.grl_alpha)
        adversarial_logits = self.adv_head(adversarial_features)
        return fingerprint_features, mask, adversarial_logits
