"""Genera muestras sintéticas desde un checkpoint entrenado (moabb_pipeline.py)
y las compara con los datos reales del sujeto: ERP overlay, PSD (Welch) +
divergencia espectral (JSD), MMD² distribucional, y PCA/t-SNE.

Reutiliza eeggan.generate_samples_main (generación) y
eeggan.helpers.visualize_pca.visualization_dim_reduction (PCA/t-SNE) — no
reimplementa nada que el paquete ya resuelva.

Estilo de gráficos y métricas (JSD, MMD²) inspirados en
/home/dvfer/Documents/GANNTHESIS/gann/EEG-DDGAN/signal_analysis.ipynb.

Uso:
    python compare_samples.py --subjects 1 2 3
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # sin display en el servidor de entrenamiento
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from scipy.spatial.distance import jensenshannon

from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.visualize_pca import visualization_dim_reduction

DATA_DIR  = 'subject_data/train'
GAN_DIR   = 'trained_models'
GEN_DIR   = 'generated_samples'
PLOT_DIR  = 'comparison_plots'
MODEL_PREFIX = 'GAN_009_Modded'

CONDITIONS = {'NonTarget': 0, 'Target': 1}
MAX_SAMPLES_PER_COND = 200  # cap para no generar batches gigantes
FS = 256  # sampling rate de BNCI2014_009; ajustar si el pipeline resamplea
BP_HZ = 24     # banda real del paradigm P300 de MOABB es [1,24] Hz
PSD_FMAX = 25  # límite de vista del PSD (referencia: signal_analysis.ipynb)

# Estilo inspirado en signal_analysis.ipynb (figuras estilo paper)
COLORS = {'real': '#0072BD', 'gen': '#D65F5F'}
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _generate_condition_csv(model_path, out_csv, condition, n_samples):
    from eeggan.generate_samples_main import main as generate_main
    n_samples = min(n_samples, MAX_SAMPLES_PER_COND)
    generate_main([
        f'model={model_path}',
        f'save_name={out_csv}',
        f'conditions={condition}',
        f'num_samples_total={n_samples}',
        f'num_samples_parallel={n_samples}',
    ])


def generate_synthetic(model_path, real_csv, out_csv):
    """Genera tantas muestras sintéticas como trials reales por condición
    (hasta MAX_SAMPLES_PER_COND) y las junta en un único CSV con el mismo
    formato largo que el real."""
    real_df = pd.read_csv(real_csv)
    n_trials_per_cond = real_df.groupby('Condition')['Trial'].nunique()

    os.makedirs(GEN_DIR, exist_ok=True)
    parts = []
    for name, cond in CONDITIONS.items():
        n = int(n_trials_per_cond.get(cond, 0))
        if n == 0:
            continue
        tmp_csv = os.path.join(GEN_DIR, f'_tmp_{name}.csv')
        _generate_condition_csv(model_path, tmp_csv, cond, n)
        parts.append(pd.read_csv(tmp_csv))
        os.remove(tmp_csv)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def load(csv_path, norm_data):
    dl = Dataloader(path=csv_path, norm_data=norm_data,
                     kw_time='Time', kw_conditions='Condition', kw_channel='Electrode')
    data = dl.get_data(shuffle=False)[:, 1:].numpy()       # (trial, seq, channel)
    labels = dl.get_labels()[:, 0, 0].numpy()              # (trial,)
    return data, labels


def _long_df(data_2d, tipo_senal):
    """(trials, tiempo) -> formato largo para sns.lineplot (promedio+banda automáticos)."""
    n_trials, seq_len = data_2d.shape
    return pd.DataFrame({
        'tiempo': np.tile(np.arange(seq_len), n_trials),
        'amplitud': data_2d.flatten(),
        'tipo_senal': tipo_senal,
    })


def plot_erp_overlay(real_data, real_labels, gen_data, gen_labels, out_path):
    n_channels = real_data.shape[-1]
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))

    for name, cond in CONDITIONS.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
        fig.suptitle(f'ERP promedio ± desvío estándar por canal — {name} (real vs sintético)')
        for ch in range(n_channels):
            ax = axs[ch // ncols, ch % ncols]
            df = pd.concat([
                _long_df(real_data[real_labels == cond][:, :, ch], 'Real'),
                _long_df(gen_data[gen_labels == cond][:, :, ch], 'Sintético'),
            ], ignore_index=True)
            sns.lineplot(data=df, x='tiempo', y='amplitud', hue='tipo_senal',
                         palette={'Real': COLORS['real'], 'Sintético': COLORS['gen']},
                         errorbar='sd', ax=ax, legend=(ch == 0))
            ax.set_title(f'Canal {ch}', fontsize=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
        if n_channels:
            axs[0, 0].legend(fontsize=7)
        for ch in range(n_channels, nrows * ncols):
            axs[ch // ncols, ch % ncols].axis('off')
        fig.tight_layout()
        path = out_path.format(cond=name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Guardado: {path}')


def spectral_jsd(freqs, psd_real, psd_gen, fmax=BP_HZ):
    """Divergencia de Jensen-Shannon entre dos PSD, restringida a la banda real
    [0, fmax] Hz. 0 = idénticas, 1 = máximamente distintas. Ref: signal_analysis.ipynb."""
    mask = freqs <= fmax
    p = psd_real[mask] + 1e-12
    q = psd_gen[mask] + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return float(jensenshannon(p, q) ** 2)


def plot_psd_overlay(real_data, real_labels, gen_data, gen_labels, out_path, fs=FS):
    """PSD (Welch) promedio por canal/condición, real vs sintético.
    Devuelve {(condición, canal): jsd} -- divergencia espectral en banda real."""
    n_channels = real_data.shape[-1]
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))
    jsd_scores = {}

    for name, cond in CONDITIONS.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
        fig.suptitle(f'PSD (Welch) por canal — {name} (real vs sintético)')
        for ch in range(n_channels):
            ax = axs[ch // ncols, ch % ncols]
            means = {}
            for label, data, labels, color, style in [
                ('Real', real_data, real_labels, COLORS['real'], '-'),
                ('Sintético', gen_data, gen_labels, COLORS['gen'], '--'),
            ]:
                trials = data[labels == cond][:, :, ch]
                if trials.shape[0] == 0:
                    continue
                nperseg = min(fs // 4, trials.shape[1])  # fs//4: mismo Welch que signal_analysis.ipynb
                f, pxx = welch(trials, fs=fs, nperseg=nperseg, axis=1)
                mean, std = pxx.mean(axis=0), pxx.std(axis=0)
                means[label] = mean
                ax.plot(f, mean, color=color, linestyle=style, label=label)
                ax.fill_between(f, mean - std, mean + std, color=color, alpha=0.2)
            if 'Real' in means and 'Sintético' in means:
                jsd_scores[(name, ch)] = spectral_jsd(f, means['Real'], means['Sintético'])
            ax.axvline(BP_HZ, color='gray', linewidth=1, linestyle=':', label=f'Corte real ({BP_HZ} Hz)')
            ax.set_xlim(0, PSD_FMAX)
            ax.set_title(f'Canal {ch}', fontsize=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
        if n_channels:
            axs[0, 0].legend(fontsize=7)
        for ch in range(n_channels, nrows * ncols):
            axs[ch // ncols, ch % ncols].axis('off')
        fig.supxlabel('Frequency (Hz)', fontsize=9)
        fig.supylabel('Power Spectral Density (a.u.)', fontsize=9)
        fig.tight_layout()
        path = out_path.format(cond=name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Guardado: {path}')

    return jsd_scores


def mmd_rbf(X, Y, n_sub=400, seed=42):
    """MMD^2 no sesgado con kernel RBF (sigma por heurística de mediana).
    Sub-samplea a n_sub para velocidad. Ref: signal_analysis.ipynb."""
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


def compare_subject(subject):
    real_csv  = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
    model_pt  = os.path.join(GAN_DIR, f'{MODEL_PREFIX}_s{subject:03d}.pt')
    gen_csv   = os.path.join(GEN_DIR, f'{MODEL_PREFIX}_s{subject:03d}_synthetic.csv')

    if not os.path.exists(real_csv):
        print(f'  Aviso: no existe {real_csv}, se omite sujeto {subject}.')
        return
    if not os.path.exists(model_pt):
        print(f'  Aviso: no existe {model_pt}, se omite sujeto {subject}.')
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f'  Generando muestras sintéticas -> {gen_csv}')
    generate_synthetic(model_pt, real_csv, gen_csv)

    real_data, real_labels = load(real_csv, norm_data=True)
    gen_data, gen_labels   = load(gen_csv, norm_data=False)  # ya está en la escala normalizada del generador

    print('  Graficando ERP promedio (real vs sintético)...')
    plot_erp_overlay(real_data, real_labels, gen_data, gen_labels,
                      os.path.join(PLOT_DIR, f's{subject:03d}_erp_{{cond}}.png'))

    print('  Graficando PSD (Welch)...')
    jsd_scores = plot_psd_overlay(real_data, real_labels, gen_data, gen_labels,
                                   os.path.join(PLOT_DIR, f's{subject:03d}_psd_{{cond}}.png'))

    print(f'  Métricas de fidelidad espectral/distribucional (sujeto {subject}):')
    for name, cond in CONDITIONS.items():
        cond_jsd = [v for (n, _ch), v in jsd_scores.items() if n == name]
        if cond_jsd:
            print(f'    JSD espectral (0-{BP_HZ}Hz) {name}: media={np.mean(cond_jsd):.4f} '
                  f'(por canal: {[round(v, 3) for v in cond_jsd]})')

        real_c = real_data[real_labels == cond].reshape(-1, real_data.shape[1] * real_data.shape[2])
        gen_c  = gen_data[gen_labels == cond].reshape(-1, gen_data.shape[1] * gen_data.shape[2])
        if len(real_c) < 4 or len(gen_c) < 4:
            continue
        half = len(real_c) // 2
        baseline = mmd_rbf(real_c[:half], real_c[half:])
        mmd_val = mmd_rbf(real_c, gen_c)
        print(f'    MMD² {name}: real-vs-sintético={mmd_val:.6f}  (baseline real/real split={baseline:.6f}, esperado ~0)')

    print('  Graficando PCA...')
    visualization_dim_reduction(real_data, gen_data, 'pca', save=True,
                                 save_name=os.path.join(PLOT_DIR, f's{subject:03d}_pca.png'))

    print('  Graficando t-SNE...')
    visualization_dim_reduction(real_data, gen_data, 'tsne', save=True,
                                 save_name=os.path.join(PLOT_DIR, f's{subject:03d}_tsne.png'))


def main():
    parser = argparse.ArgumentParser(description='Compara muestras generadas por la GAN con los datos reales.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()

    for subject in args.subjects:
        print(f'\n{"="*55}\n  Sujeto {subject}\n{"="*55}')
        compare_subject(subject)

    print('\nComparación completada. Ver PNGs en', PLOT_DIR)


if __name__ == '__main__':
    main()
