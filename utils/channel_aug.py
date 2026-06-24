import torch


def add_random_awgn(signal, snr_range=(0.0, 30.0)):
    """Add sample-wise AWGN with SNR drawn uniformly from ``snr_range``."""
    min_snr, max_snr = map(float, snr_range)
    if min_snr > max_snr:
        raise ValueError("The minimum SNR cannot exceed the maximum SNR.")
    snr = torch.empty(signal.shape[0], device=signal.device, dtype=signal.dtype)
    snr.uniform_(min_snr, max_snr)
    return add_awgn(signal, snr)


def add_awgn(signal, snr):
    """Add AWGN to each sample while preserving its independently measured power."""
    if signal.ndim < 2:
        raise ValueError("signal must include a batch dimension and at least one feature dimension")

    reduce_dims = tuple(range(1, signal.ndim))
    signal_power = torch.mean(signal.square(), dim=reduce_dims, keepdim=True)
    snr = torch.as_tensor(snr, device=signal.device, dtype=signal.dtype)
    if snr.ndim == 0:
        snr = snr.repeat(signal.shape[0])
    if snr.numel() != signal.shape[0]:
        raise ValueError("snr must be a scalar or contain one value per sample")
    snr = snr.reshape(-1, *([1] * (signal.ndim - 1)))
    noise_power = signal_power / torch.pow(signal.new_tensor(10.0), snr / 10.0)
    return signal + torch.randn_like(signal) * torch.sqrt(noise_power.clamp_min(1e-12))
