#!/usr/bin/env bash
# Corre moabb_pipeline.py (modelo ttsgan-direct: GAN directa sobre EEG
# multicanal crudo, DWT + stacking, sin autoencoder) para los sujetos que
# faltan del BNCI2014_009 (10 sujetos, subject 1 ya corrido).
#
# moabb_pipeline.py ya acepta --subjects y ya salta la exportación de CSV si
# el train/test ya existe, pero SIEMPRE reentrena la GAN aunque el .pt ya
# exista, y un solo sujeto que tire excepción mata el proceso entero (mata el
# resto de la lista). Por eso lo llamamos un sujeto a la vez acá: así un
# sujeto que falla no corta a los demás, y podés relanzar el script para
# retomar (se salta los .pt que ya existen).
#
# Uso:
#   ./run_remaining_subjects.sh              # sujetos 2..10
#   ./run_remaining_subjects.sh 4 5 6         # sujetos específicos

set -uo pipefail

if [[ $# -gt 0 ]]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(2 3 4 5 6 7 8 9 10)
fi

GAN_DIR="trained_models"
# Debe matchear el MODEL_PREFIX calculado en moabb_pipeline.py (fm/postnet/stack).
MODEL_PREFIX="GAN_009_fm50_postnet1_stack0"
LOG_DIR="logs/remaining_subjects_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILED=()
for s in "${SUBJECTS[@]}"; do
    ckpt="${GAN_DIR}/${MODEL_PREFIX}_s$(printf '%03d' "$s").pt"
    if [[ -f "$ckpt" ]]; then
        echo "=== Sujeto $s: ya existe $ckpt, se omite ==="
        continue
    fi
    echo "=== Sujeto $s: entrenando (log en ${LOG_DIR}/subject_${s}.log) ==="
    if python moabb_pipeline.py --subjects "$s" 2>&1 | tee "${LOG_DIR}/subject_${s}.log"; then
        echo "=== Sujeto $s: OK ==="
    else
        echo "=== Sujeto $s: FALLÓ, ver ${LOG_DIR}/subject_${s}.log ===" >&2
        FAILED+=("$s")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "" >&2
    echo "Sujetos fallidos: ${FAILED[*]}" >&2
    exit 1
fi
echo "Listo: todos los sujetos entrenados."
