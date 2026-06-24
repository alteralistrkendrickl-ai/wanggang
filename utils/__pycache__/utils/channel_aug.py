import torch
import random


def add_random_awgn(signal, snr_range=(0, 30)):
    """
    向信号添加随机信噪比的 AWGN 噪声
    
    Args:
        signal: 输入信号张量，形状为 (batch, channels, length) 或类似
        snr_range: 信噪比范围 (min_snr, max_snr)，单位为 dB
    
    Returns:
        添加噪声后的信号
    """
    if snr_range[0] == snr_range[1]:
        snr = snr_range[0]
    else:
        snr = random.uniform(snr_range[0], snr_range[1])
    
    return add_awgn(signal, snr)


def add_awgn(signal, snr):
    """
    向信号添加指定信噪比的 AWGN 噪声
    
    Args:
        signal: 输入信号张量
        snr: 信噪比，单位为 dB
    
    Returns:
        添加噪声后的信号
    """
    # 计算信号功率
    signal_power = torch.mean(signal ** 2)
    
    # 根据 SNR 计算噪声功率
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr / 10))
    
    # 生成高斯噪声
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    
    # 添加噪声
    noisy_signal = signal + noise
    
    return noisy_signal
