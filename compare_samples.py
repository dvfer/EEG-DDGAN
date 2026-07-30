"""Genera muestras sintéticas desde un checkpoint entrenado (moabb_pipeline.py)
y las compara visualmente con los datos reales del sujeto: ERP promediado por
canal/condición (overlay) + PCA/t-SNE.

Reutiliza eeggan.generate_samples_main (generación) y
eeggan.helpers.visualize_pca.visualization_dim_reduction (PCA/t-SNE) — no
reimplementa nada que el paquete ya resuelva.

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

from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.visualize_pca import visualization_dim_reduction

DATA_DIR  = 'subject_data/train'
GAN_DIR   = 'trained_models'
GEN_DIR   = 'generated_samples'
PLOT_DIR  = 'comparison_plots'
MODEL_PREFIX = 'GAN_009_Modded'

CONDITIONS = {'NonTarget': 0, 'Target': 1}
MAX_SAMPLES_PER_COND = 200  # cap para no generar batches gigantes


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


def plot_erp_overlay(real_data, real_labels, gen_data, gen_labels, out_path):
    n_channels = real_data.shape[-1]
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))

    for name, cond in CONDITIONS.items():
        real_avg = real_data[real_labels == cond].mean(axis=0)
        gen_avg  = gen_data[gen_labels == cond].mean(axis=0)

        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
        fig.suptitle(f'ERP promedio por canal — {name} (real vs sintético)')
        for ch in range(n_channels):
            ax = axs[ch // ncols, ch % ncols]
            ax.plot(real_avg[:, ch], color='black', label='Real')
            ax.plot(gen_avg[:, ch], color='red', linestyle='--', label='Sintético')
            ax.set_title(f'Canal {ch}', fontsize=8)
        axs[0, 0].legend(fontsize=7)
        for ch in range(n_channels, nrows * ncols):
            axs[ch // ncols, ch % ncols].axis('off')
        fig.tight_layout()
        path = out_path.format(cond=name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Guardado: {path}')


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
