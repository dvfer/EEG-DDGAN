#!/usr/bin/env bash
# Corre train_eegnet_augmentation.py un sujeto a la vez (aísla fallas -- p.ej.
# falta el checkpoint DDGAN de ese sujeto) y deja logs.
#
# Uso:
#   ./train_eegnet_augmentation.sh          # sujetos 1..10
#   ./train_eegnet_augmentation.sh 4 5 6    # sujetos específicos

set -uo pipefail

if [[ $# -gt 0 ]]; then
    SUBJECTS=("$@")
else
    SUBJECTS=(1 2 3 4 5 6 7 8 9 10)
fi

LOG_DIR="logs/eegnet_augmentation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FAILED=()
for s in "${SUBJECTS[@]}"; do
    echo "=== Sujeto $s: corriendo (log en ${LOG_DIR}/subject_${s}.log) ==="
    if uv run python train_eegnet_augmentation.py "$s" 2>&1 | tee "${LOG_DIR}/subject_${s}.log"; then
        echo "=== Sujeto $s: OK ==="
    else
        echo "=== Sujeto $s: FALLÓ, ver ${LOG_DIR}/subject_${s}.log ===" >&2
        FAILED+=("$s")
    fi
done

echo ""
echo "=== Tabla combinada ==="
cat ablation_results/eegnet_augmentation_*.csv 2>/dev/null

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "" >&2
    echo "Sujetos fallidos: ${FAILED[*]}" >&2
    exit 1
fi
echo "Listo: todos los sujetos entrenados y evaluados."
