#!/usr/bin/env bash
# Corre train_eeggan_vanilla.sh (EEG-GAN "vanilla" de `main`, AE-coupled,
# target=full -- el que mejores resultados dio) para varios sujetos, uno a la
# vez -- mismo motivo que run_remaining_subjects.sh: un sujeto que falla no
# debe matar el resto de la lista, y podés relanzar el script para retomar
# (train_eeggan_vanilla.sh ya salta el AE/GAN cuyo .pt exista).
#
# Después de cada sujeto, agrega sus muestras sintéticas a la tabla de
# ablation vía eval_external_config.py (ya calcula JSD/MMD²/error pico ERP
# contra el held-out de test de ESE sujeto).
#
# Uso:
#   ./train_eeggan_full_all_subjects.sh          # sujetos 1..10
#   ./train_eeggan_full_all_subjects.sh 4 5 6    # sujetos específicos

set -uo pipefail

if [[ $# -gt 0 ]]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(1 2 3 4 5 6 7 8 9 10)
fi

TARGET="full"
CONFIG_NAME="eeg_gan_vanilla_${TARGET}"
LOG_DIR="logs/eeggan_${TARGET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILED=()
for s in "${SUBJECTS[@]}"; do
    s_fmt=$(printf "%03d" "$s")
    gen_csv="generated_samples/EEG_GAN_vanilla_${TARGET}_s${s_fmt}_synthetic.csv"

    echo "=== Sujeto $s: entrenando (log en ${LOG_DIR}/subject_${s}.log) ==="
    if ! ./train_eeggan_vanilla.sh "$s" "$TARGET" 2>&1 | tee "${LOG_DIR}/subject_${s}.log"; then
        echo "=== Sujeto $s: FALLÓ el entrenamiento, ver ${LOG_DIR}/subject_${s}.log ===" >&2
        FAILED+=("$s")
        continue
    fi

    echo "=== Sujeto $s: agregando a la tabla de ablation ==="
    if ! uv run python eval_external_config.py --name "$CONFIG_NAME" --gen-csv "$gen_csv" --subject "$s" \
            2>&1 | tee -a "${LOG_DIR}/subject_${s}.log"; then
        echo "=== Sujeto $s: FALLÓ la evaluación, ver ${LOG_DIR}/subject_${s}.log ===" >&2
        FAILED+=("$s")
    fi
done

SUMMARY_CSV="ablation_results/metrics_${CONFIG_NAME}.csv"
uv run python -c "
import glob, pandas as pd
rows = []
for f in glob.glob('ablation_results/ablation_s*.csv'):
    df = pd.read_csv(f)
    row = df[df['config'] == '${CONFIG_NAME}']
    if not row.empty:
        subject = int(f.split('_s')[-1].split('.')[0])
        rows.append({'subject': subject, **row.iloc[0].drop('config').to_dict()})
if rows:
    out = pd.DataFrame(rows).sort_values('subject')
    out.to_csv('${SUMMARY_CSV}', index=False)
    print(f'\nTabla combinada -> ${SUMMARY_CSV}')
    print(out.to_string(index=False))
else:
    print('Ningún sujeto tiene fila ${CONFIG_NAME} en ablation_results/.')
"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "" >&2
    echo "Sujetos fallidos: ${FAILED[*]}" >&2
    exit 1
fi
echo "Listo: todos los sujetos entrenados y evaluados (target=${TARGET})."
