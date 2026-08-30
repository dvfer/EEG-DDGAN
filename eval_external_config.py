"""Agrega un modelo entrenado FUERA de esta rama (ej. EEG-GAN vanilla de `main`,
o el tts-gan de referencia) a la tabla comparativa de ablation_pipeline.py.

No entrena ni genera nada acá — ese checkpoint vive en otra rama/repo con un
formato incompatible con esta (generate_samples_main.py de ttsgan-direct
rechaza checkpoints con autoencoder). Este script solo:
  1. toma el CSV de muestras sintéticas ya generado ALLÁ (con el
     generate_samples de esa otra rama/repo)
  2. calcula las mismas métricas (compare_subject: JSD, MMD², pico ERP) y
     los mismos plots (ERP/PSD/PCA/t-SNE) que el resto de las configs
  3. agrega/actualiza su fila en ablation_results/ablation_s<subject>.csv
  4. re-grafica el PSD combinado incluyéndolo

⚠️  El CSV externo debe estar en el mismo formato largo que Dataloader espera
(ParticipantID, Condition, Trial, Electrode, Time1..TimeN) y en una escala
comparable a los datos reales (acá se cargan con norm_data=True). Si la
normalización de esa otra rama/repo es distinta, JSD sigue siendo válida
(compara la FORMA de la PSD, normalizada a que sume 1) pero MMD² y el error
de amplitud del pico ERP sí son sensibles a la escala absoluta — revisa que
los rangos de valores sean comparables antes de confiar en esos dos.

Uso:
    python eval_external_config.py --name eeg_gan_vanilla --gen-csv path/a/muestras.csv
"""
import argparse
import os
import shutil

import pandas as pd

from ablation_pipeline import ABLATIONS, ABLATION_SUBJECT, RESULTS_DIR, gen_csv_path, plot_combined_psd
from compare_samples import compare_subject, DATA_DIR, TEST_DATA_DIR


def add_external_config(name, external_gen_csv, subject=ABLATION_SUBJECT):
    train_csv = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
    real_csv = os.path.join(TEST_DATA_DIR, f'subject_{subject:03d}.csv')
    if not os.path.exists(real_csv):
        raise FileNotFoundError(f'No existe {real_csv}')
    if not os.path.exists(external_gen_csv):
        raise FileNotFoundError(f'No existe {external_gen_csv}')

    target_csv = gen_csv_path(name, subject=subject)
    os.makedirs(os.path.dirname(target_csv), exist_ok=True)
    shutil.copy(external_gen_csv, target_csv)

    metrics = compare_subject(
        subject,
        real_csv=real_csv,
        train_csv=train_csv,
        gen_csv=target_csv,
        plot_dir=os.path.join(RESULTS_DIR, name),
        skip_generation=True,
    )
    if metrics is None:
        raise RuntimeError(f"No se pudo evaluar '{name}' — revisa el CSV de entrada.")

    csv_path = os.path.join(RESULTS_DIR, f'ablation_s{subject:03d}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['config'] != name]  # reemplaza la fila si ya existía
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([{'config': name, **metrics}])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f'\nTabla actualizada -> {csv_path}')
    print(df.to_string(index=False))

    already_run = [cfg['name'] for cfg in ABLATIONS if os.path.exists(gen_csv_path(cfg['name'], subject=subject))]
    plot_combined_psd(real_csv, train_csv=train_csv, config_names=already_run + [name], subject=subject)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help="Nombre de la config, ej. 'eeg_gan_vanilla'")
    parser.add_argument('--gen-csv', required=True, help='CSV de muestras sintéticas ya generado en otra rama/repo')
    parser.add_argument('--subject', type=int, default=ABLATION_SUBJECT)
    args = parser.parse_args()
    add_external_config(args.name, args.gen_csv, args.subject)
