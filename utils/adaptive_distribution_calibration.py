"""Lightweight base-to-novel distribution calibration in feature space."""

import numpy as np


METHODS = ("lr", "mean", "diag", "gated")


def compute_class_statistics(features, labels, variance_floor=1e-4):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must align")
    classes = np.unique(labels).astype(int)
    means, variances = [], []
    for class_id in classes:
        current = features[labels == class_id]
        means.append(current.mean(axis=0))
        variance = current.var(axis=0, ddof=1) if len(current) > 1 else np.zeros(features.shape[1])
        variances.append(np.maximum(variance, variance_floor))
    return classes, np.asarray(means, dtype=np.float32), np.asarray(variances, dtype=np.float32)


def nearest_base_statistics(target_mean, base_means, base_variances, top_m=3):
    if top_m < 1 or top_m > len(base_means):
        raise ValueError("top_m must be between 1 and the number of base classes")
    target = np.asarray(target_mean, dtype=np.float32)
    base_means = np.asarray(base_means, dtype=np.float32)
    base_variances = np.asarray(base_variances, dtype=np.float32)
    target_norm = target / max(float(np.linalg.norm(target)), 1e-12)
    base_norm = base_means / np.maximum(np.linalg.norm(base_means, axis=1, keepdims=True), 1e-12)
    nearest = np.argsort(-(base_norm @ target_norm))[:top_m]
    return base_means[nearest].mean(axis=0), base_variances[nearest].mean(axis=0), nearest


def shot_gate(shot, scale=5.0):
    if shot < 1 or scale <= 0:
        raise ValueError("shot and scale must be positive")
    return float(scale / (scale + shot))


def augment_support_features(
    support_features,
    support_labels,
    base_means,
    base_variances,
    method,
    rng,
    top_m=3,
    mean_strength=0.2,
    variance_strength=0.2,
    gate_scale=5.0,
    synthetic_per_class=50,
    variance_floor=1e-4,
):
    """Return real support plus calibrated synthetic features for one method."""
    if method not in METHODS:
        raise ValueError(f"unknown A2 method: {method}")
    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).reshape(-1)
    if method == "lr":
        return features.copy(), labels.copy()
    if synthetic_per_class < 1:
        raise ValueError("synthetic_per_class must be positive")

    generated_features = [features]
    generated_labels = [labels]
    for class_id in np.unique(labels).astype(int):
        current = features[labels == class_id]
        shot = len(current)
        target_mean = current.mean(axis=0)
        target_variance = (
            current.var(axis=0, ddof=1) if shot > 1 else np.full(features.shape[1], variance_floor)
        )
        reference_mean, reference_variance, _ = nearest_base_statistics(
            target_mean, base_means, base_variances, top_m=top_m
        )
        gate = shot_gate(shot, gate_scale) if method == "gated" else 1.0
        generated_count = max(1, int(np.ceil(synthetic_per_class * gate)))
        alpha = mean_strength * gate
        calibrated_mean = (1.0 - alpha) * target_mean + alpha * reference_mean

        if method == "mean":
            indices = rng.integers(0, shot, size=generated_count)
            synthetic = current[indices] + (calibrated_mean - target_mean)
        else:
            beta = variance_strength * gate
            calibrated_variance = (
                (1.0 - beta) * np.maximum(target_variance, variance_floor)
                + beta * np.maximum(reference_variance, variance_floor)
            )
            synthetic = rng.normal(
                calibrated_mean,
                np.sqrt(np.maximum(calibrated_variance, variance_floor)),
                size=(generated_count, features.shape[1]),
            )
        generated_features.append(np.asarray(synthetic, dtype=np.float32))
        generated_labels.append(np.full(generated_count, class_id, dtype=labels.dtype))
    return np.concatenate(generated_features), np.concatenate(generated_labels)
