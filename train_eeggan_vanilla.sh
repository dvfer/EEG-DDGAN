#!/usr/bin/env bash
# Entrena y genera muestras con el EEG-GAN "vanilla" (rama `main` del fork, con
# autoencoder, sin DWT/PostNet/stacking) sobre el MISMO CSV de train que ya
# exportó moabb_pipeline.py, para compararlo en la tabla de ablation_pipeline.py
# vía eval_external_config.py.
#
# Receta calcada de train_autoencoders.sh/train_vanilla_gan.sh (ya probados
# funcionando): AE con target=time (NO 'full') + GAN condicional directo sobre
# ese mismo AE. target=time usa TransformerAutoencoder (un solo nivel, opera
# sobre el eje temporal) en vez de TransformerDoubleAutoencoder -- esta última
# es la que tiene una capa Linear fija al conteo de canales, que rompe cuando
# la GAN condicional le pega un canal extra de condición encima (ver
# conversación: mismatch 16 vs 17 canales). target=time no tiene ese problema.
#
# Usa `main` del fork (no el upstream real AutoResearch/EEG-GAN): el upstream
# puro tiene el orden de EncoderDiscriminator.forward() invertido/roto
# (discrimina la señal cruda y DESPUÉS intenta "encodear" la salida del
# discriminador -- no tiene sentido y explota), mientras que `main` del fork
# ya lo tiene arreglado (encode primero, discrimina después). Corre `main` en
# un git worktree aparte -- no toca tu working tree de ttsgan-direct.
#
# Uso:
#   ./train_eeggan_vanilla.sh [subject_id]   # default: 1

set -euo pipefail

# ── CONFIGURACIÓN (ajustar antes de correr) ─────────────────────────────
SUBJECT="${1:-1}"
SUBJECT_FMT=$(printf "%03d" "$SUBJECT")

# Presupuesto de entrenamiento -- alinealo con ABLATION_N_EPOCHS de
# ablation_pipeline.py si querés que la comparación sea de presupuesto parejo.
N_EPOCHS_AE=2000
N_EPOCHS_GAN=2000
SEED=42

# Autoencoder: target=time (NO full -- ver nota arriba), igual a
# train_autoencoders.sh
TIME_OUT=50
PATCH_SIZE=10

# "_time" en el nombre: si alguna vez cambiás target= acá, un checkpoint
# viejo (ej. entrenado con target=full) no se reusa en silencio con la
# config nueva -- ver el mismatch de canales que documentamos arriba.
AE_NAME="AE_vanilla_time_s${SUBJECT_FMT}"
GAN_NAME="EEG_GAN_vanilla_time_s${SUBJECT_FMT}"

WORKTREE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../EEG-DDGAN-main-vanilla"
# ─────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRAIN_CSV="${REPO_ROOT}/subject_data/train/subject_${SUBJECT_FMT}.csv"
TEST_CSV="${REPO_ROOT}/subject_data/test/subject_${SUBJECT_FMT}.csv"

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "No existe $TRAIN_CSV -- corré moabb_pipeline.py --subjects $SUBJECT primero (en ttsgan-direct)." >&2
    exit 1
fi
if [[ ! -f "$TEST_CSV" ]]; then
    echo "No existe $TEST_CSV -- corré moabb_pipeline.py --subjects $SUBJECT primero (en ttsgan-direct)." >&2
    exit 1
fi

echo "=== Preparando worktree de 'main' en $WORKTREE_DIR ==="
if [[ ! -d "$WORKTREE_DIR" ]]; then
    git worktree add "$WORKTREE_DIR" main
else
    echo "  Worktree ya existe, se reusa."
fi

cd "$WORKTREE_DIR"
# uv sync/run harían resolución "universal" respetando el requires-python del
# pyproject.toml (>=3.7), que no matchea con pandas>=2.2 (necesita >=3.9) --
# uv venv + uv pip install resuelven solo para el intérprete concreto.
# --python 3.11: torch==2.3.1 (lo que fija eeggan) no tiene wheels para 3.13+.
# Si un intento previo dejó un .venv roto: rm -rf .venv
[[ -d .venv ]] || uv venv --python 3.11
uv pip install -e . --python .venv
# main's pyproject.toml no tiene el índice CUDA (cu128) que sí tiene
# ttsgan-direct -- sin esto cae al wheel default de PyPI, sin kernels para
# GPUs nuevas ("no kernel image is available for execution on the device").
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7,<3.0" "torchvision>=0.22,<1.0" "torchaudio>=2.7,<3.0" --python .venv
# pytorch_wavelets: dependencia no declarada en pyproject.toml de main (ver
# CLAUDE.md) -- models.py la importa a nivel de módulo sin condicionarla.
# pkg_resources (setuptools<81, ver_a>=81 la sacó) y PyWavelets son sus
# propias dependencias transitivas, tampoco declaradas por pytorch_wavelets.
uv pip install "setuptools<81" pytorch_wavelets PyWavelets --python .venv
PY=".venv/bin/python"

AE_CKPT="trained_ae/${AE_NAME}.pt"
GAN_CKPT="trained_models/${GAN_NAME}.pt"

# Invocamos eeggan.*_main.main() directamente (como ya hace moabb_pipeline.py/
# compare_samples.py) en vez de `python -m eeggan`, que importa TODOS los
# comandos a nivel de módulo apenas se lo carga (arrastra deps que ni usamos).

echo ""
echo "=== 1. Autoencoder (target=time, time_out=${TIME_OUT}) ==="
if [[ -f "$AE_CKPT" ]]; then
    echo "  Ya existe $AE_CKPT, se omite entrenamiento."
else
    "$PY" - <<PYEOF
from eeggan.autoencoder_training_main import main
main([
    "data=$TRAIN_CSV",
    "kw_channel=Electrode",
    "target=time",
    "time_out=$TIME_OUT",
    "n_epochs=$N_EPOCHS_AE",
    "seed=$SEED",
    "save_name=$AE_NAME",
])
PYEOF
fi

echo ""
echo "=== 2. GAN vanilla condicional (AE-coupled, sin DWT/PostNet/stacking) ==="
if [[ -f "$GAN_CKPT" ]]; then
    echo "  Ya existe $GAN_CKPT, se omite entrenamiento."
else
    "$PY" - <<PYEOF
from eeggan.gan_training_main import main
main([
    "data=$TRAIN_CSV",
    "kw_channel=Electrode",
    "kw_conditions=Condition",
    "patch_size=$PATCH_SIZE",
    "n_epochs=$N_EPOCHS_GAN",
    "seed=$SEED",
    "autoencoder=$AE_CKPT",
    "save_name=$GAN_NAME",
])
PYEOF
fi

echo ""
echo "=== 3. Generando muestras sintéticas (tantas como trials reales de test por condición) ==="
mkdir -p generated_samples
N_TARGET=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TEST_CSV'); print(df[df.Condition==1]['Trial'].nunique())")
N_NONTARGET=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TEST_CSV'); print(df[df.Condition==0]['Trial'].nunique())")
echo "  Test set real: Target=${N_TARGET}, NonTarget=${N_NONTARGET}"

"$PY" - <<PYEOF
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)  # ver nota: generate_samples_main.py de main no lo fija, a diferencia de gan_training_main.py; torch>=2.6 cambio el default
from eeggan.generate_samples_main import main
main([
    "model=$GAN_CKPT",
    "save_name=_tmp_target.csv",
    "conditions=1",
    "num_samples_total=$N_TARGET",
    "num_samples_parallel=$N_TARGET",
    "sequence_length=-1",
])
PYEOF

"$PY" - <<PYEOF
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)  # ver nota: generate_samples_main.py de main no lo fija, a diferencia de gan_training_main.py; torch>=2.6 cambio el default
from eeggan.generate_samples_main import main
main([
    "model=$GAN_CKPT",
    "save_name=_tmp_nontarget.csv",
    "conditions=0",
    "num_samples_total=$N_NONTARGET",
    "num_samples_parallel=$N_NONTARGET",
    "sequence_length=-1",
])
PYEOF

OUT_CSV="${REPO_ROOT}/generated_samples/EEG_GAN_vanilla_s${SUBJECT_FMT}_synthetic.csv"
mkdir -p "${REPO_ROOT}/generated_samples"
"$PY" -c "
import pandas as pd
df = pd.concat([pd.read_csv('generated_samples/_tmp_target.csv'), pd.read_csv('generated_samples/_tmp_nontarget.csv')], ignore_index=True)
df.to_csv('$OUT_CSV', index=False)
print(f'  Guardado: $OUT_CSV ({len(df)} filas)')
"
rm -f generated_samples/_tmp_target.csv generated_samples/_tmp_nontarget.csv

echo ""
echo "=== Listo ==="
echo "Para agregarlo a la tabla de ablation (desde el worktree de ttsgan-direct):"
echo "  cd $REPO_ROOT"
echo "  python eval_external_config.py --name eeg_gan_vanilla --gen-csv generated_samples/EEG_GAN_vanilla_s${SUBJECT_FMT}_synthetic.csv --subject $SUBJECT"
