#!/usr/bin/env bash
# Entrena train_tts_gan_baseline.py un sujeto a la vez (mismo motivo que
# run_remaining_subjects.sh: un sujeto que tira excepción no debe matar el
# resto de la lista) y deja logs + tabla de métricas incremental.
#
# Uso:
#   ./train_tts_gan_baseline.sh          # sujetos 1..10
#   ./train_tts_gan_baseline.sh 4 5 6    # sujetos específicos

set -uo pipefail

if [[ $# -gt 0 ]]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(1 2 3 4 5 6 7 8 9 10)
fi

LOG_DIR="logs/tts_gan_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILED=()
for s in "${SUBJECTS[@]}"; do
    echo "=== Sujeto $s: corriendo (log en ${LOG_DIR}/subject_${s}.log) ==="
    if uv run python train_tts_gan_baseline.py "$s" 2>&1 | tee "${LOG_DIR}/subject_${s}.log"; then
        echo "=== Sujeto $s: OK ==="
    else
        echo "=== Sujeto $s: FALLÓ, ver ${LOG_DIR}/subject_${s}.log ===" >&2
        FAILED+=("$s")
    fi
done

echo ""
echo "=== Tabla combinada: ablation_results/metrics_tts_gan_baseline.csv ==="
cat ablation_results/metrics_tts_gan_baseline.csv 2>/dev/null

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "" >&2
    echo "Sujetos fallidos: ${FAILED[*]}" >&2
    exit 1
fi
echo "Listo: todos los sujetos entrenados y evaluados."
