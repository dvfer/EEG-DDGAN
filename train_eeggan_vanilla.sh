#!/usr/bin/env bash
# Entrena y genera muestras con el EEG-GAN "vanilla" (rama `main`, con autoencoder,
# sin DWT/PostNet/stacking) sobre el MISMO CSV de train que ya exportó
# moabb_pipeline.py en ttsgan-direct, para poder compararlo en la tabla de
# ablation_pipeline.py vía eval_external_config.py.
#
# `main` no tiene moabb_pipeline.py (fue agregado en mv-train/ttsgan-direct), así
# que reusa el CSV de train ya exportado en vez de re-descargar/exportar de MOABB.
# Corre `main` en un git worktree aparte -- no toca tu working tree de ttsgan-direct.
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

# Autoencoder (target=full: recomendado para datos multicanal, ver CLAUDE.md)
CHANNELS_OUT=10
TIME_OUT=50
PATCH_SIZE=10   # debe dividir a TIME_OUT (el generador corre en espacio del AE)

AE_NAME="AE_vanilla_s${SUBJECT_FMT}"
GAN_NAME="EEG_GAN_vanilla_s${SUBJECT_FMT}"

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
# uv sync/run harían resolución "universal" respetando el requires-python de
# main's pyproject.toml (>=3.7), que no matchea con pandas>=2.2 (necesita
# >=3.9) -- explota buscando compatibilidad con 3.7/3.8 que ni usamos.
# uv venv + uv pip install resuelven solo para el intérprete concreto, evita
# ese choque sin tocar el pyproject.toml de main.
# --python 3.11: torch==2.3.1 (el que fija eeggan==2.0.2) no tiene wheels
# para 3.13+; uv descarga/gestiona el 3.11 solo, no depende del python del
# sistema/conda. Si un intento previo dejó un .venv roto, borralo primero:
#   rm -rf .venv
[[ -d .venv ]] || uv venv --python 3.11
uv pip install -e . --python .venv
# main's pyproject.toml no tiene el [tool.uv.sources]/[tool.uv.index] que sí
# tiene ttsgan-direct apuntando torch al build CUDA (cu128) -- sin esto cae
# al wheel default de PyPI, que no trae kernels para GPUs nuevas ("no kernel
# image is available for execution on the device"). Forzamos el mismo índice
# que ya sabemos que funciona en esta máquina.
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7,<3.0" "torchvision>=0.22,<1.0" "torchaudio>=2.7,<3.0" --python .venv
# pytorch_wavelets: dependencia no declarada en pyproject.toml (ver CLAUDE.md)
# -- models.py la importa a nivel de módulo sin condicionarla, hace falta
# aunque no se use el discriminador DWT. A su vez usa pkg_resources
# (setuptools), que uv venv no incluye por defecto (a diferencia de un venv
# armado con pip clásico). setuptools>=81 sacó pkg_resources (deprecado
# desde 2025-11-30) -- hay que fijar una versión anterior. PyWavelets
# (pywt) es a su vez una dependencia de pytorch_wavelets que su propio
# paquete no declara.
uv pip install "setuptools<81" pytorch_wavelets PyWavelets --python .venv
PY=".venv/bin/python"

AE_CKPT="trained_ae/${AE_NAME}.pt"
GAN_CKPT="trained_models/${GAN_NAME}.pt"

# Invocamos cada eeggan.*_main.main() directamente (como ya hace
# moabb_pipeline.py/compare_samples.py) en vez de `python -m eeggan`:
# eeggan/__main__.py importa TODOS los comandos a nivel de módulo apenas se
# lo carga, así que arrastra dependencias de comandos que ni usamos (ej.
# `requests`, que solo necesita setup_tutorial) sin necesidad real.

echo ""
echo "=== 1. Autoencoder (target=full, channels_out=${CHANNELS_OUT}, time_out=${TIME_OUT}) ==="
if [[ -f "$AE_CKPT" ]]; then
    echo "  Ya existe $AE_CKPT, se omite entrenamiento."
else
    "$PY" - <<PYEOF
from eeggan.autoencoder_training_main import main
main([
    "data=$TRAIN_CSV",
    "kw_channel=Electrode",
    "kw_time=Time",
    "target=full",
    "channels_out=$CHANNELS_OUT",
    "time_out=$TIME_OUT",
    "n_epochs=$N_EPOCHS_AE",
    "seed=$SEED",
    "save_name=$AE_NAME",
])
PYEOF
fi

echo ""
echo "=== 2. GAN vanilla (AE-coupled, sin DWT/PostNet/stacking) ==="
if [[ -f "$GAN_CKPT" ]]; then
    echo "  Ya existe $GAN_CKPT, se omite entrenamiento."
else
    "$PY" - <<PYEOF
from eeggan.gan_training_main import main
main([
    "data=$TRAIN_CSV",
    "kw_channel=Electrode",
    "kw_conditions=Condition",
    "kw_time=Time",
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
