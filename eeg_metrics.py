"""Métricas de fidelidad señal-a-señal (real vs. sintético), compartidas por
compare_samples.py y ablation_pipeline.py — una sola fuente de verdad para no
repetir la lógica de Welch/JSD/MMD²/ERP en dos scripts.

Metodología alineada con
/home/dvfer/Documents/GANNTHESIS/gann/EEG-DDGAN/signal_analysis.ipynb
(nperseg = fs // 4, banda JSD = banda real del paradigm, ventana P300 =
(250, 600) ms tomada de su constante P300_WIN).
"""

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon


def compute_psd(trials, fs):
    """PSD (Welch) por trial, promediada. trials: (n_trials, seq_len).
    Devuelve (freqs, mean, std). nperseg = fs//4, igual que signal_analysis.ipynb."""
    nperseg = min(fs // 4, trials.shape[1])
    f, pxx = welch(trials, fs=fs, nperseg=nperseg, axis=1)
    return f, pxx.mean(axis=0), pxx.std(axis=0)


def spectral_jsd(freqs, psd_real, psd_gen, fmax):
    """Divergencia de Jensen-Shannon entre dos PSD, restringida a [0, fmax] Hz.
    0 = idénticas, 1 = máximamente distintas."""
    mask = freqs <= fmax
    p = psd_real[mask] + 1e-12
    q = psd_gen[mask] + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return float(jensenshannon(p, q) ** 2)


def mmd_rbf(X, Y, n_sub=400, seed=42):
    """MMD^2 no sesgado con kernel RBF (sigma por heurística de mediana).
    Sub-samplea a n_sub para velocidad."""
    rng = np.random.default_rng(seed)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    if len(Y) > n_sub:
        Y = Y[rng.choice(len(Y), n_sub, replace=False)]

    combined = np.vstack([X, Y])
    sq_dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
    sigma2 = np.median(sq_dists[sq_dists > 0])

    def rbf(A, B):
        d2 = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * sigma2))

    kxx, kyy, kxy = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


def _erp_peak(trials, fs, window_ms):
    """Pico (amplitud+latencia) del ERP promedio dentro de una ventana.
    trials: (n_trials, seq_len). Igual a erp_peak() de signal_analysis.ipynb."""
    seq_len = trials.shape[1]
    t_ms = np.linspace(0, seq_len / fs * 1000, seq_len)
    win_mask = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
    grand_avg = trials.mean(axis=0)
    win_signal = grand_avg[win_mask]
    peak_idx = np.argmax(win_signal)
    return t_ms[win_mask][peak_idx], win_signal[peak_idx]


def erp_peak_metrics(real_trials, gen_trials, fs, window_ms=(250, 600)):
    """Error de amplitud/latencia del pico P300 (real vs sintético).
    Ventana por defecto (250, 600) ms: P300_WIN de signal_analysis.ipynb.
    Devuelve (amp_error, latency_error_ms)."""
    real_lat, real_amp = _erp_peak(real_trials, fs, window_ms)
    gen_lat, gen_amp = _erp_peak(gen_trials, fs, window_ms)
    return abs(gen_amp - real_amp), abs(gen_lat - real_lat)


def _demo_self_check():
    """ponytail: smallest thing that fails if the metric math breaks."""
    rng = np.random.default_rng(0)
    fs = 256
    trials = rng.normal(size=(20, 128)).astype(np.float32)

    f, mean, std = compute_psd(trials, fs)
    assert f.shape == mean.shape == std.shape
    assert spectral_jsd(f, mean, mean, fmax=24) < 1e-9   # identical PSD -> ~0 JSD
    assert spectral_jsd(f, mean, mean + 10, fmax=24) > spectral_jsd(f, mean, mean, fmax=24)

    assert mmd_rbf(trials, trials, seed=1) < 1e-6         # identical set -> ~0 MMD²
    other = rng.normal(loc=5, size=(20, 128)).astype(np.float32)
    assert mmd_rbf(trials, other, seed=1) > mmd_rbf(trials, trials, seed=1)

    # inject a known P300-like bump so the peak-finder has a ground truth to hit
    t_ms = np.linspace(0, trials.shape[1] / fs * 1000, trials.shape[1])
    bump = 3.0 * np.exp(-0.5 * ((t_ms - 400) / 30) ** 2)
    gen = trials + bump
    amp_err, lat_err = erp_peak_metrics(trials, gen, fs)
    assert lat_err < 50   # detected peak should land near the 400ms bump we injected
    assert amp_err > 0

    print("eeg_metrics self-check: OK")


if __name__ == '__main__':
    _demo_self_check()
