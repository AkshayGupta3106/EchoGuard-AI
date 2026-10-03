"""
phone_channel.py -- simulate a narrow-band phone channel on 16 kHz mono float audio.

    band-pass 300-3400 Hz -> 8 kHz -> mu-law (8-bit) round trip -> back to 16 kHz -> + noise (SNR 25 dB)

Use it to check whether a spoof detector still separates TTS from real speech
once both have been through a phone-like channel. It is a rough model (no AMR/AAC
codec, no packet loss), so treat results as indicative only.
"""
import numpy as np
import scipy.signal as ss


def phone_channel(x: np.ndarray, snr_db: float = 25.0, seed: int = 0) -> np.ndarray:
    b, a = ss.butter(4, [300 / 8000, 3400 / 8000], "band")
    y = ss.filtfilt(b, a, x)
    y8 = ss.resample_poly(y, 1, 2)
    mu = 255.0
    comp = np.sign(y8) * np.log1p(mu * np.abs(np.clip(y8, -1, 1))) / np.log1p(mu)
    q = np.round(comp * 127) / 127
    dec = np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)
    y16 = ss.resample_poly(dec, 2, 1)
    rms = np.sqrt((y16 ** 2).mean()) + 1e-9
    noise = np.random.RandomState(seed).randn(len(y16)) * rms * 10 ** (-snr_db / 20)
    return (y16 + noise).astype(np.float32)
