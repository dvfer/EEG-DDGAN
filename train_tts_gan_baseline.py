"""Entrena el TTS-GAN "puro" (baseline: sin DWT/stacking/PostNet — el config
'baseline' de ablation_pipeline.py, pero a presupuesto de producción en vez
del recorte ABLATION_N_EPOCHS=200) por sujeto y reporta sus métricas contra
el held-out de test. Reusa train_gan()/compare_subject() ya existentes — no
reimplementa entrenamiento ni evaluación.

Guarda su .pt aparte de ABLATION_baseline_sXXX.pt (que quedó a 200 epochs
para iterar rápido) para no confundir un checkpoint corto con el de reporte
final.

Uso:
    uv run python train_tts_gan_baseline.py            # sujetos 1..10
    uv run python train_tts_gan_baseline.py 4 5 6       # sujetos específicos
"""
import os
import sys

import pandas as pd

from compare_samples import compare_subject, DATA_DIR, TEST_DATA_DIR, GAN_DIR, GEN_DIR, PLOT_DIR
from moabb_pipeline import train_gan, GAN_PATCH_SIZE, GAN_N_EPOCHS, GAN_SEED

NAME = 'tts_gan_baseline'
RESULTS_CSV = 'ablation_results/metrics_tts_gan_baseline.csv'


def checkpoint_path(subject):
    return os.path.join(GAN_DIR, f'GAN_009_{NAME}_s{subject:03d}.pt')


def gen_csv_path(subject):
    return os.path.join(GEN_DIR, f'GAN_009_{NAME}_s{subject:03d}_synthetic.csv')


def run_subject(subject):
    train_csv = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
    test_csv = os.path.join(TEST_DATA_DIR, f'subject_{subject:03d}.csv')
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"Falta {train_csv} y/o {test_csv} — corré moabb_pipeline.py "
            f"--subjects {subject} primero."
        )

    ckpt = checkpoint_path(subject)
    if os.path.exists(ckpt):
        print(f'  Checkpoint ya existe, se omite entrenamiento: {ckpt}')
    else:
        train_gan(
            train_csv, ckpt,
            patch_size=GAN_PATCH_SIZE,
            n_epochs=GAN_N_EPOCHS,
            seed=GAN_SEED,
            use_dwt=False,
            use_stacking=False,
            use_postnet=False,
        )

    return compare_subject(
        subject,
        model_path=ckpt,
        real_csv=test_csv,
        train_csv=train_csv,
        gen_csv=gen_csv_path(subject),
        plot_dir=os.path.join(PLOT_DIR, NAME),
    )


if __name__ == '__main__':
    subjects = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
    os.makedirs('ablation_results', exist_ok=True)

    rows = []
    if os.path.exists(RESULTS_CSV):
        rows = pd.read_csv(RESULTS_CSV).to_dict('records')
        rows = [r for r in rows if int(r['subject']) not in subjects]  # se recalculan

    for subject in subjects:
        print(f'\n{"="*55}\n  Sujeto {subject} ({NAME})\n{"="*55}')
        metrics = run_subject(subject)
        if metrics is None:
            print(f'  Aviso: sujeto {subject} omitido (falta checkpoint o test CSV).')
            continue
        rows.append({'subject': subject, **metrics})
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)  # guarda progreso incremental

    if not rows:
        print('\nNingún sujeto produjo métricas.')
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values('subject')
    df.to_csv(RESULTS_CSV, index=False)
    print(f'\nTabla -> {RESULTS_CSV}')
    print(df.to_string(index=False))
