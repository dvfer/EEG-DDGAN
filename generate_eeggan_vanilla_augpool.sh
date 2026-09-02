#!/usr/bin/env bash
# Genera un pool sintético de EEG-GAN vanilla (rama `main`, AE-coupled) sizeado
# contra TRAIN (no contra test, a diferencia de train_eeggan_vanilla.sh, que lo
# genera sizeado a test solo para la tabla de ablation) -- lo necesita
# train_eegnet_augmentation.py para poder armar ratios de aumento de hasta 0.5
# de train. Requiere que train_eeggan_vanilla.sh ya haya entrenado ese sujeto
# (no entrena nada acá, solo genera).
#
# Cap de 200 trials por condición (misma convención que
# compare_samples.MAX_SAMPLES_PER_COND) para no generar pools gigantes.
#
# Uso:
#   ./generate_eeggan_vanilla_augpool.sh [subject_id] [target]   # default: 1 full

set -euo pipefail

SUBJECT="${1:-1}"
SUBJECT_FMT=$(printf "%03d" "$SUBJECT")
TARGET="${2:-full}"
MAX_PER_COND=200

GAN_NAME="EEG_GAN_vanilla_${TARGET}_s${SUBJECT_FMT}"
WORKTREE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../EEG-DDGAN-main-vanilla"

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRAIN_CSV="${REPO_ROOT}/subject_data/train/subject_${SUBJECT_FMT}.csv"
OUT_CSV="${REPO_ROOT}/generated_samples/${GAN_NAME}_augpool.csv"

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "No existe $TRAIN_CSV -- corré moabb_pipeline.py --subjects $SUBJECT primero." >&2
    exit 1
fi
if [[ ! -d "$WORKTREE_DIR" ]]; then
    echo "No existe $WORKTREE_DIR -- corré ./train_eeggan_vanilla.sh $SUBJECT $TARGET primero." >&2
    exit 1
fi

cd "$WORKTREE_DIR"
PY=".venv/bin/python"
GAN_CKPT="trained_models/${GAN_NAME}.pt"
if [[ ! -f "$GAN_CKPT" ]]; then
    echo "No existe $GAN_CKPT -- corré ./train_eeggan_vanilla.sh $SUBJECT $TARGET primero." >&2
    exit 1
fi

N_TARGET=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TRAIN_CSV'); print(min(df[df.Condition==1]['Trial'].nunique(), $MAX_PER_COND))")
N_NONTARGET=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TRAIN_CSV'); print(min(df[df.Condition==0]['Trial'].nunique(), $MAX_PER_COND))")
echo "Generando pool: Target=${N_TARGET}, NonTarget=${N_NONTARGET}"

mkdir -p generated_samples
for cond_pair in "1 target $N_TARGET" "0 nontarget $N_NONTARGET"; do
    read -r cond name n <<< "$cond_pair"
    "$PY" - <<PYEOF
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)

import eeggan.helpers.initialize_gan as _ig
_orig_init_gan = _ig.init_gan
def _init_gan_2tuple(*a, **kw):
    result = _orig_init_gan(*a, **kw)
    return result[:2] if len(result) > 2 else result
_ig.init_gan = _init_gan_2tuple

from eeggan.generate_samples_main import main
main([
    "model=$GAN_CKPT",
    "save_name=_tmp_${name}.csv",
    "conditions=$cond",
    "num_samples_total=$n",
    "num_samples_parallel=$n",
    "sequence_length=-1",
])
PYEOF
done

mkdir -p "${REPO_ROOT}/generated_samples"
"$PY" -c "
import pandas as pd
df = pd.concat([pd.read_csv('generated_samples/_tmp_target.csv'), pd.read_csv('generated_samples/_tmp_nontarget.csv')], ignore_index=True)
df.to_csv('$OUT_CSV', index=False)
print(f'  Guardado: $OUT_CSV ({len(df)} filas)')
"
rm -f generated_samples/_tmp_target.csv generated_samples/_tmp_nontarget.csv
