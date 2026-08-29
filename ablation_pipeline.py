"""Ablation study: impacto del segundo discriminador (DWT/stacking), el PostNet
y el peso de feature-matching (lambda_fm) en la calidad de la señal sintética.

Entrena la matriz de configs ABLATIONS (todas sobre el mismo sujeto/CSV/patch_size/
seed/n_epochs, solo cambia el knob ablado) reusando train_gan() de moabb_pipeline.py,
y evalúa cada checkpoint reusando compare_subject() de compare_samples.py (que ya
genera muestras + ERP/PSD/PCA/t-SNE + JSD/MMD²/ERP-peak) — ningún código de
entrenamiento/generación/métricas se reimplementa aquí.

Ver openspec/changes/ablation-study-harness/ (proposal/design/specs/tasks).

Uso:
    python ablation_pipeline.py
"""

import os

import numpy as np
import pandas as pd

from compare_samples import compare_subject, load, DATA_DIR, GAN_DIR, GEN_DIR, FS, PSD_FMAX
from eeg_metrics import compute_psd
from moabb_pipeline import train_gan

# ────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────────────────

ABLATION_SUBJECT   = 1
ABLATION_N_EPOCHS  = 200   # ponytail: bajo a propósito para iterar rápido; subir a
                           # escala de producción (ver GAN_N_EPOCHS en moabb_pipeline.py)
                           # para los números finales de la tesis
ABLATION_PATCH_SIZE = 10
ABLATION_SEED        = 42

RESULTS_DIR = 'ablation_results'

# 9 configs: baseline, PostNet solo, DWT (con/sin stacking), y el factorial
# lambda_fm x use_postnet (ver openspec design.md D3 para el razonamiento de cada una)
ABLATIONS = [
    {'name': 'baseline',             'use_dwt': False, 'use_stacking': False, 'use_postnet': False},
    {'name': 'postnet_only',         'use_dwt': False, 'use_stacking': False, 'use_postnet': True},
    {'name': 'dwt',                  'use_dwt': True,  'use_stacking': False, 'use_postnet': False, 'lambda_fm': 20},
    {'name': 'dwt_stacking',         'use_dwt': True,  'use_stacking': True,  'use_postnet': False, 'lambda_fm': 20},
    {'name': 'fm_lambda_0',          'use_dwt': True,  'use_stacking': False, 'use_postnet': False, 'lambda_fm': 0},
    {'name': 'fm_lambda_20_postnet', 'use_dwt': True,  'use_stacking': False, 'use_postnet': True,  'lambda_fm': 20},  # == modelo completo
    {'name': 'fm_lambda_50',         'use_dwt': True,  'use_stacking': False, 'use_postnet': False, 'lambda_fm': 50},
    {'name': 'fm_lambda_0_postnet',  'use_dwt': True,  'use_stacking': False, 'use_postnet': True,  'lambda_fm': 0},
    {'name': 'fm_lambda_50_postnet', 'use_dwt': True,  'use_stacking': False, 'use_postnet': True,  'lambda_fm': 50},
]

# ────────────────────────────────────────────────────────────


def checkpoint_path(name):
    return os.path.join(GAN_DIR, f'ABLATION_{name}_s{ABLATION_SUBJECT:03d}.pt')


def gen_csv_path(name):
    return os.path.join(GEN_DIR, f'ABLATION_{name}_s{ABLATION_SUBJECT:03d}_synthetic.csv')


def plot_combined_psd(real_csv):
    """Una figura: PSD real + las 9 configs superpuestas, por canal."""
    import matplotlib.pyplot as plt

    real_data, _ = load(real_csv, norm_data=True)
    n_channels = real_data.shape[-1]
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    fig.suptitle(f'PSD comparado — configs de ablation, sujeto {ABLATION_SUBJECT}')

    cmap = plt.get_cmap('tab10')
    for ch in range(n_channels):
        ax = axs[ch // ncols, ch % ncols]
        f, mean, _ = compute_psd(real_data[:, :, ch], FS)
        ax.plot(f, mean, color='black', linewidth=1.5, label='Real')
        ax.set_xlim(0, PSD_FMAX)
        ax.set_title(f'Canal {ch}', fontsize=8)

    for i, cfg in enumerate(ABLATIONS):
        gcsv = gen_csv_path(cfg['name'])
        if not os.path.exists(gcsv):
            continue
        gen_data, _ = load(gcsv, norm_data=False)
        for ch in range(n_channels):
            ax = axs[ch // ncols, ch % ncols]
            f, mean, _ = compute_psd(gen_data[:, :, ch], FS)
            ax.plot(f, mean, color=cmap(i % 10), linewidth=1, alpha=0.85, label=cfg['name'])

    if n_channels:
        axs[0, 0].legend(fontsize=6, ncol=2)
    for ch in range(n_channels, nrows * ncols):
        axs[ch // ncols, ch % ncols].axis('off')
    fig.supxlabel('Frequency (Hz)', fontsize=9)
    fig.supylabel('Power Spectral Density (a.u.)', fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f'psd_comparison_s{ABLATION_SUBJECT:03d}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Guardado: {out_path}')


def run_ablation():
    real_csv = os.path.join(DATA_DIR, f'subject_{ABLATION_SUBJECT:03d}.csv')
    if not os.path.exists(real_csv):
        raise FileNotFoundError(
            f"No existe {real_csv} — corre moabb_pipeline.py con "
            f"--subjects {ABLATION_SUBJECT} primero."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for cfg in ABLATIONS:
        name = cfg['name']
        ckpt = checkpoint_path(name)
        print(f'\n{"="*55}\n  Config: {name} ({cfg})\n{"="*55}')

        if os.path.exists(ckpt):
            print(f'  Checkpoint ya existe, se omite entrenamiento: {ckpt}')
        else:
            train_gan(
                real_csv, ckpt,
                patch_size=ABLATION_PATCH_SIZE,
                n_epochs=ABLATION_N_EPOCHS,
                seed=ABLATION_SEED,
                use_dwt=cfg.get('use_dwt', False),
                high_freq=True,
                lambda_fm=cfg.get('lambda_fm', 20),
                use_postnet=cfg.get('use_postnet', False),
                use_stacking=cfg.get('use_stacking', False),
            )

        metrics = compare_subject(
            ABLATION_SUBJECT,
            model_path=ckpt,
            real_csv=real_csv,
            gen_csv=gen_csv_path(name),
            plot_dir=os.path.join(RESULTS_DIR, name),
        )
        if metrics is None:
            print(f"  Aviso: no se pudo evaluar '{name}', se omite de la tabla.")
            continue
        rows.append({'config': name, **metrics})

    if not rows:
        print('\nNinguna config produjo métricas — nada que tabular.')
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f'ablation_s{ABLATION_SUBJECT:03d}.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTabla combinada -> {csv_path}')
    print(df.to_string(index=False))

    print('\nGraficando PSD combinado...')
    plot_combined_psd(real_csv)

    print('\nAblation completado. Ver', RESULTS_DIR)


if __name__ == '__main__':
    run_ablation()
