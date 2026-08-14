"""Device-safe STC metric objective for the adapted FS-SEI baseline."""

import torch
from torch import nn


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=5.0):
        super().__init__()
        self.ranking = nn.MarginRankingLoss(margin=float(margin))

    def forward(self, embeddings, labels):
        distances = torch.cdist(embeddings, embeddings, p=2)
        same = labels[:, None].eq(labels[None, :])
        different = ~same
        if not different.any():
            # A deterministic validation loader may end with a single-class
            # remainder.  It contributes no valid triplet instead of making
            # the whole validation epoch fail.
            return embeddings.sum() * 0.0
        hardest_positive = distances.masked_fill(~same, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~different, float("inf")).min(dim=1).values
        targets = torch.ones_like(hardest_negative)
        return self.ranking(hardest_negative, hardest_positive, targets)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, embeddings, labels):
        selected = self.centers.index_select(0, labels)
        return (embeddings - selected).pow(2).sum(dim=1).mean()


class FSSEISTCLoss(nn.Module):
    """CE + 0.01 batch-hard triplet + 0.01 center loss."""

    def __init__(
        self,
        num_classes=90,
        feature_dim=1024,
        weights=(1.0, 0.01, 0.01),
        triplet_margin=5.0,
    ):
        super().__init__()
        if len(weights) != 3 or any(weight < 0 for weight in weights):
            raise ValueError("weights must contain three non-negative values")
        self.weights = tuple(float(weight) for weight in weights)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.triplet = BatchHardTripletLoss(triplet_margin)
        self.center = CenterLoss(num_classes, feature_dim)

    def forward(self, logits, embeddings, labels):
        ce = self.cross_entropy(logits, labels)
        triplet = self.triplet(embeddings, labels)
        center = self.center(embeddings, labels)
        total = self.weights[0] * ce + self.weights[1] * triplet + self.weights[2] * center
        return total, ce, triplet, center


def create_model(**kwargs):
    return FSSEISTCLoss(**kwargs)
