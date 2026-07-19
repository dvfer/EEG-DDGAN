"""Pipeline: BNCI2014_009 (P300) → CSV multi-canal → GAN (ttsgan-direct)

Reemplaza los bucles bash de entrenamiento. Llama directamente a las
funciones Python de eeggan sin subprocesses.

Nota (rama ttsgan-direct): esta rama entrena la GAN directamente sobre los
datos multicanal en crudo, sin el paso de autoencoder que usaba `main`
(ver openspec/changes/ttsgan-native-multichannel). `patch_size` debe ser
divisor de la longitud de secuencia cruda del dataset (ya no de `time_out`
del AE) — `gan_training_main.py` lanza un `ValueError` claro si no lo es.

Uso:
    python moabb_pipeline.py

Ajusta las variables de configuración de abajo antes de ejecutar.
"""

import argparse
import os
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# CONFIGURACIÓN  (equivalente a las variables del script bash)
# ────────────────────────────────────────────────────────────

# Sujetos a procesar (IDs de BNCI2014_009: 1-10)
SUBJECTS = [1, 2, 3]

# Condición a exportar: 'Target', 'NonTarget', o 'Both'
CONDITION = 'Both'

# Normalización z-score por trial antes de exportar (False = sin normalizar)
NORM = False

# Directorios de salida
DATA_DIR = 'subject_data/train'
GAN_DIR  = 'trained_models'

# Prefijo para los archivos de modelo
MODEL_PREFIX = 'GAN_009_Modded'

# ── GAN ─────────────────────────────────────────────────────
GAN_PATCH_SIZE = 10
GAN_N_EPOCHS   = 2000
GAN_SEED       = 42
GAN_USE_DWT    = True   # usar MultiscaleDWTDiscriminator
GAN_HIGH_FREQ  = True   # incluir coeficientes de alta frecuencia en DWT
GAN_USE_STACKING = True # combinar D1 (TTS) + D2 (DWT) vía StackingDiscriminator

# ────────────────────────────────────────────────────────────


# Mapeo de etiquetas string (P300) → entero
_P300_LABEL_MAP = {'Target': 1, 'NonTarget': 0}


def _to_int_labels(y):
    """Convierte etiquetas string ('Target'/'NonTarget') a int si es necesario."""
    y = np.asarray(y)
    if y.dtype.kind in ('U', 'S', 'O'):            # dtype string/object
        return np.vectorize(_P300_LABEL_MAP.get)(y).astype(int)
    return y.astype(int)


def get_channel_names(dataset, subject):
    """Extrae nombres de canales EEG desde los datos raw MNE del sujeto."""
    try:
        raws = dataset.get_data(subjects=[subject])
        # estructura: {subject: {session: {run: MNE_Raw}}}
        first_subject = next(iter(raws.values()))
        first_session = next(iter(first_subject.values()))
        first_run     = next(iter(first_session.values()))
        picks = first_run.pick('eeg', verbose=False)
        return list(picks.ch_names)
    except Exception as exc:
        print(f"  Aviso: no se pudieron extraer nombres de canales ({exc}). "
              f"Usando Ch0, Ch1, ...")
        return None   # se determina más tarde a partir de X.shape


def export_to_csv(X, y, ch_names, condition='Both', output_path='data.csv',
                  norm=False):
    """Exporta EEG multi-canal a CSV en formato largo compatible con Dataloader.

    Args:
        X:          numpy array (n_trials, n_channels, n_timepoints)
        y:          array-like (n_trials,) — etiquetas int o str ('Target'/'NonTarget')
        ch_names:   list[str] de longitud n_channels
        condition:  'Target' (solo clase 1), 'NonTarget' (clase 0), o 'Both'
        output_path: ruta de salida del CSV
        norm:       si True, aplica z-score por trial antes de exportar

    CSV resultante (formato largo):
        ParticipantID | Condition | Trial | Electrode | Time1 | Time2 | ...
        una fila por (trial, canal)

    Uso posterior con Dataloader:
        Dataloader(output_path, kw_channel='Electrode',
                   kw_time='Time', kw_conditions='Condition')
    """
    X = np.asarray(X, dtype=np.float32)
    y = _to_int_labels(y)

    if X.ndim != 3:
        raise ValueError(
            f"X debe ser 3-D (n_trials, n_channels, n_timepoints), "
            f"recibido {X.shape}"
        )

    n_trials, n_channels, n_timepoints = X.shape

    if ch_names is None:
        ch_names = [f'Ch{i}' for i in range(n_channels)]
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names tiene {len(ch_names)} entradas pero X tiene {n_channels} canales"
        )

    # Filtrar por condición
    if condition == 'Target':
        mask = y == 1
    elif condition == 'NonTarget':
        mask = y == 0
    else:
        mask = np.ones(n_trials, dtype=bool)

    X_out = X[mask]
    y_out = y[mask]
    n_export = X_out.shape[0]

    if n_export == 0:
        raise ValueError(
            f"Sin datos para exportar con condition='{condition}'. "
            f"Etiquetas únicas presentes: {np.unique(y)}"
        )

    # Normalización z-score por trial (sobre todos los canales y tiempo)
    if norm:
        mean = X_out.mean(axis=(1, 2), keepdims=True)
        std  = X_out.std(axis=(1, 2), keepdims=True) + 1e-8
        X_out = (X_out - mean) / std

    # Shuffle si exportamos ambas condiciones
    if condition == 'Both':
        perm   = np.random.permutation(n_export)
        X_out  = X_out[perm]
        y_out  = y_out[perm]

    # Construir DataFrame en formato largo (una fila por canal por trial)
    # Nota: la columna Electrode guarda el ÍNDICE del canal (entero) en vez del
    # nombre, para que todo el CSV sea numérico y compatible con el Dataloader
    # (df.to_numpy() debe ser float, no object). El mapeo índice→nombre se
    # guarda en un archivo sidecar '<csv>.channels.txt' para remapear después.
    time_cols = [f'Time{t + 1}' for t in range(n_timepoints)]
    rows = []
    for trial_idx in range(n_export):
        for ch_idx in range(n_channels):
            row = {
                'ParticipantID': 1,
                'Condition':     int(y_out[trial_idx]),
                'Trial':         trial_idx,
                'Electrode':     ch_idx,
            }
            row.update(zip(time_cols, X_out[trial_idx, ch_idx].tolist()))
            rows.append(row)

    col_order = ['ParticipantID', 'Condition', 'Trial', 'Electrode'] + time_cols
    df = pd.DataFrame(rows, columns=col_order)
    df.to_csv(output_path, index=False)

    # Sidecar con el mapeo índice→nombre de canal (un nombre por línea)
    channels_path = f'{output_path}.channels.txt'
    with open(channels_path, 'w') as f:
        f.write('\n'.join(ch_names) + '\n')

    print(
        f"  CSV exportado: {output_path}\n"
        f"  Channels map:  {channels_path}\n"
        f"  Trials: {n_export} | Canales: {n_channels} | "
        f"Timepoints: {n_timepoints} | Filas totales: {len(df)}"
    )
    return df


def train_gan(csv_path, gan_save_path, patch_size=10,
              n_epochs=2000, seed=42, use_dwt=True, high_freq=True, use_stacking=False):
    """Entrena la GAN llamando directamente a eeggan (sin autoencoder — ttsgan-direct)."""
    from eeggan.gan_training_main import main as gan_main

    print(f"  Entrenando GAN → {gan_save_path}")
    args = [
        f'data={csv_path}',
        f'save_name={os.path.basename(gan_save_path)}',
        'kw_channel=Electrode',
        'kw_conditions=Condition',
        'kw_time=Time',
        f'patch_size={patch_size}',
        f'n_epochs={n_epochs}',
        f'seed={seed}',
    ]
    if use_dwt:
        args.append('use_multiscale_dwt_discriminator')
    if high_freq:
        args.append('multiscale_dwt_high_freq=True')
    if use_stacking:
        args.append('use_stacking')
    gan_main(args)


def main():
    from moabb.datasets import BNCI2014_009
    from moabb.paradigms import P300

    parser = argparse.ArgumentParser(
        description="Pipeline BNCI2014_009 (P300) -> CSV -> GAN (ttsgan-direct, sin autoencoder)"
    )
    parser.add_argument(
        '--subjects', type=int, nargs='+', default=SUBJECTS,
        help='IDs de sujetos a procesar (default: SUBJECTS del archivo). Ej: --subjects 1'
    )
    args = parser.parse_args()

    dataset  = BNCI2014_009()
    paradigm = P300()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(GAN_DIR,  exist_ok=True)

    for subject in args.subjects:
        print(f'\n{"="*55}')
        print(f'  Sujeto {subject}')
        print(f'{"="*55}')

        # Rutas de archivos
        csv_path      = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
        gan_save_path = os.path.join(GAN_DIR,  f'{MODEL_PREFIX}_s{subject:03d}.pt')

        # 1. Exportar CSV (se omite si ya existe)
        if os.path.exists(csv_path):
            print(f'  CSV ya existe, se omite la exportación: {csv_path}')
        else:
            print('  Cargando datos MOABB...')
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            ch_names = get_channel_names(dataset, subject)

            n_ch = X.shape[1]
            if ch_names is None:
                ch_names = [f'Ch{i}' for i in range(n_ch)]

            print(f'  Shape: {X.shape} | Canales: {ch_names[:5]}{"..." if n_ch > 5 else ""}')
            print('  Exportando CSV...')
            export_to_csv(
                X, y, ch_names,
                condition=CONDITION,
                output_path=csv_path,
                norm=NORM,
            )

        # 2. Entrenar GAN directamente sobre los datos crudos multicanal (sin AE)
        train_gan(
            csv_path, gan_save_path,
            patch_size=GAN_PATCH_SIZE,
            n_epochs=GAN_N_EPOCHS,
            seed=GAN_SEED,
            use_dwt=GAN_USE_DWT,
            high_freq=GAN_HIGH_FREQ,
            use_stacking=GAN_USE_STACKING,
        )

    print('\nPipeline completado.')


if __name__ == '__main__':
    main()
