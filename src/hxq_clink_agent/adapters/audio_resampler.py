"""音频重采样工具 - 基于 numpy 线性插值的 PCM 采样率转换.

用于在天润融通 8kHz PCM 与百度 RTC 16kHz/24kHz PCM 之间进行转换。
"""

import numpy as np


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """将 PCM 音频从源采样率重采样到目标采样率.

    使用线性插值算法，输入输出均为 16bit signed LE PCM。

    Args:
        pcm: 原始 PCM 字节数据（16bit signed LE）
        src_rate: 源采样率（如 8000）
        dst_rate: 目标采样率（如 16000 或 24000）

    Returns:
        重采样后的 PCM 字节数据（16bit signed LE）
    """
    if src_rate == dst_rate:
        return pcm

    if len(pcm) == 0:
        return pcm

    # bytes → int16 numpy array
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)

    # 计算重采样后的样本数
    n_src = len(samples)
    n_dst = int(round(n_src * dst_rate / src_rate))
    if n_dst == 0:
        return b""

    # 线性插值
    src_indices = np.linspace(0, n_src - 1, n_src)
    dst_indices = np.linspace(0, n_src - 1, n_dst)
    resampled = np.interp(dst_indices, src_indices, samples)

    # float64 → int16
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    return resampled.tobytes()
