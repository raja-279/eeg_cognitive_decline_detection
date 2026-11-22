"""
preprocessing.py  (v2)
3-channel band RP + PSD spectrogram + epoch augmentation
"""
import numpy as np
import mne
from scipy.signal import spectrogram as sp_spectrogram
from scipy.ndimage import zoom

mne.set_log_level("WARNING")

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

N_CHANNELS = {"recurrence": 3, "psd": 1}


def load_and_preprocess(set_path, l_freq=0.5, h_freq=45.0, notch_freq=50.0):
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_window="hamming", verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    return raw


def extract_epochs(raw, epoch_duration=2.0, overlap=1.0):
    sfreq = raw.info["sfreq"]
    data = raw.get_data().astype(np.float32)
    n_ch, n_total = data.shape
    epoch_samples = int(epoch_duration * sfreq)
    step_samples = max(1, int((epoch_duration - overlap) * sfreq))
    epochs = []
    start = 0
    while start + epoch_samples <= n_total:
        epochs.append(data[:, start: start + epoch_samples])
        start += step_samples
    return np.array(epochs, dtype=np.float32)


def _bandpass_fft(signal, lo, hi, sfreq):
    n = len(signal)
    fft = np.fft.rfft(signal, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    fft[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(fft, n=n).astype(np.float32)


def _rp_single(signal, img_size, threshold):
    idx = np.linspace(0, len(signal) - 1, img_size).astype(int)
    s = signal[idx]
    dist = np.abs(s[:, None] - s[None, :])
    mx = dist.max()
    dist_n = dist / mx if mx > 0 else dist
    soft = (1.0 - dist_n).astype(np.float32)
    hard = (dist_n < threshold).astype(np.float32)
    return 0.5 * soft + 0.5 * hard


def epoch_to_recurrence_plot(epoch, img_size=64, threshold=0.2, sfreq=256.0, **kwargs):
    n_ch = epoch.shape[0]
    channels = []
    for _name, (lo, hi) in BANDS.items():
        band_rps = [_rp_single(_bandpass_fft(epoch[c], lo, hi, sfreq), img_size, threshold)
                    for c in range(n_ch)]
        channels.append(np.stack(band_rps, axis=0).mean(axis=0))
    return np.stack(channels, axis=0).astype(np.float32)   # (3, H, W)


def epoch_to_psd_spectrogram(epoch, img_size=64, sfreq=256.0, **kwargs):
    n_ch = epoch.shape[0]
    specs = []
    for c in range(n_ch):
        nperseg = min(64, epoch.shape[1] // 4)
        f, _t, Sxx = sp_spectrogram(epoch[c].astype(np.float64), fs=sfreq, nperseg=nperseg)
        mask = (f >= 0.5) & (f <= 45.0)
        specs.append(np.log1p(Sxx[mask]))
    spec = np.stack(specs, axis=0).mean(axis=0).astype(np.float32)
    zh = img_size / spec.shape[0]
    zw = img_size / spec.shape[1]
    spec = zoom(spec, (zh, zw), order=1).astype(np.float32)
    mn, mx = spec.min(), spec.max()
    if mx > mn:
        spec = (spec - mn) / (mx - mn)
    return spec[np.newaxis]   # (1, H, W)


REPRESENTATION_FN = {"recurrence": epoch_to_recurrence_plot, "psd": epoch_to_psd_spectrogram}


def epoch_to_image(epoch, representation="recurrence", img_size=64, sfreq=256.0, **kwargs):
    if representation not in REPRESENTATION_FN:
        raise ValueError(f"Unknown representation '{representation}'.")
    return REPRESENTATION_FN[representation](epoch, img_size=img_size, sfreq=sfreq, **kwargs)


def augment_epoch(epoch, sfreq=None, noise_std=0.01, max_shift=10):
    epoch = epoch + (np.random.randn(*epoch.shape).astype(np.float32)
                     * noise_std * max(epoch.std(), 1e-8))
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(epoch, shift, axis=1)


def process_subject(set_path, epoch_duration=2.0, img_size=64,
                    representation="recurrence", overlap=1.0):
    raw = load_and_preprocess(set_path)
    sfreq = float(raw.info["sfreq"])
    epochs = extract_epochs(raw, epoch_duration=epoch_duration, overlap=overlap)
    images = np.stack(
        [epoch_to_image(ep, representation=representation, img_size=img_size, sfreq=sfreq)
         for ep in epochs], axis=0)
    return images
