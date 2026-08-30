#!/usr/bin/env bash
# Entrena y genera muestras con el EEG-GAN "vanilla" (autoencoder + GAN, upstream
# AutoResearch/EEG-GAN puro -- sin DWT/PostNet/stacking) sobre el MISMO CSV de
# train que ya exportó moabb_pipeline.py, para compararlo en la tabla de
# ablation_pipeline.py vía eval_external_config.py.
#
# Usa el repo upstream real (no la rama `main` del fork, que ya tiene mezclado
# código de DWT/stacking) -- clonalo antes de correr este script:
#   git clone git@github.com:AutoResearch/EEG-GAN.git ../EEG-GAN
#
# El AE-coupled discriminator de EEG-GAN no soporta condiciones (autoencoder_
# training no tiene kw_conditions -- es una limitación real del upstream, no
# del fork): un AE entrenado sin condiciones no calza con una GAN entrenada
# con kw_conditions (channel_in_disc = n_channels + n_conditions != canales
# del AE). Por eso entrenamos DOS modelos completos no-condicionales, uno por
# condición (Target/NonTarget), cada uno sobre su propio subset de trials --
# evita el mismatch por completo, y el CSV final queda igual de utilizable
# (con su columna Condition pegada a mano después de generar).
#
# Uso:
#   ./train_eeggan_vanilla.sh [subject_id]   # default: 1

set -euo pipefail

# ── CONFIGURACIÓN (ajustar antes de correr) ─────────────────────────────
SUBJECT="${1:-1}"
SUBJECT_FMT=$(printf "%03d" "$SUBJECT")

# Presupuesto de entrenamiento -- alinealo con ABLATION_N_EPOCHS de
# ablation_pipeline.py si querés que la comparación sea de presupuesto parejo.
# ojo: cada condición entrena sobre un subset más chico que el train completo
# (Target es ~1/6 de los trials) -- 2000 epochs puede sobreajustar ahí, bajalo
# si ves la loss de test despegarse mucho de la de train.
N_EPOCHS_AE=2000
N_EPOCHS_GAN=2000
SEED=42

# Autoencoder (target=full: recomendado para datos multicanal, ver CLAUDE.md)
CHANNELS_OUT=10
TIME_OUT=50
PATCH_SIZE=10   # debe dividir a TIME_OUT (el generador corre en espacio del AE)

EEG_GAN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../EEG-GAN"
# ─────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRAIN_CSV="${REPO_ROOT}/subject_data/train/subject_${SUBJECT_FMT}.csv"
TEST_CSV="${REPO_ROOT}/subject_data/test/subject_${SUBJECT_FMT}.csv"

if [[ ! -d "$EEG_GAN_REPO" ]]; then
    echo "No existe $EEG_GAN_REPO -- cloná el upstream primero:" >&2
    echo "  git clone git@github.com:AutoResearch/EEG-GAN.git $EEG_GAN_REPO" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "No existe $TRAIN_CSV -- corré moabb_pipeline.py --subjects $SUBJECT primero (en ttsgan-direct)." >&2
    exit 1
fi
if [[ ! -f "$TEST_CSV" ]]; then
    echo "No existe $TEST_CSV -- corré moabb_pipeline.py --subjects $SUBJECT primero (en ttsgan-direct)." >&2
    exit 1
fi

cd "$EEG_GAN_REPO"
echo "=== Preparando entorno en $EEG_GAN_REPO ==="
# uv sync/run harían resolución "universal" respetando el requires-python del
# pyproject.toml (>=3.7), que no matchea con pandas>=2.2 (necesita >=3.9) --
# uv venv + uv pip install resuelven solo para el intérprete concreto.
# --python 3.11: torch==2.3.1 (lo que fija eeggan) no tiene wheels para 3.13+.
# Si un intento previo dejó un .venv roto: rm -rf .venv
[[ -d .venv ]] || uv venv --python 3.11
uv pip install -e . --python .venv
# pyproject.toml no trae índice CUDA propio (a diferencia de ttsgan-direct) --
# sin esto cae al wheel default de PyPI, sin kernels para GPUs nuevas.
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7,<3.0" "torchvision>=0.22,<1.0" "torchaudio>=2.7,<3.0" --python .venv
PY=".venv/bin/python"

mkdir -p generated_samples

# Invocamos eeggan.*_main.main() directamente (como ya hace moabb_pipeline.py/
# compare_samples.py) en vez de `python -m eeggan`, que importa TODOS los
# comandos a nivel de módulo apenas se lo carga (arrastra deps que ni usamos).

for PAIR in "NonTarget:0" "Target:1"; do
    COND_NAME="${PAIR%%:*}"
    COND_VAL="${PAIR##*:}"
    echo ""
    echo "=== Condición: $COND_NAME (Condition=$COND_VAL) ==="

    COND_TRAIN_CSV="/tmp/eeggan_vanilla_train_${COND_NAME}_s${SUBJECT_FMT}.csv"
    "$PY" -c "
import pandas as pd
df = pd.read_csv('$TRAIN_CSV')
df[df['Condition'] == $COND_VAL].to_csv('$COND_TRAIN_CSV', index=False)
"

    AE_NAME="AE_vanilla_${COND_NAME}_s${SUBJECT_FMT}"
    GAN_NAME="EEG_GAN_vanilla_${COND_NAME}_s${SUBJECT_FMT}"
    AE_CKPT="trained_ae/${AE_NAME}.pt"
    GAN_CKPT="trained_models/${GAN_NAME}.pt"

    echo "  -- Autoencoder --"
    if [[ -f "$AE_CKPT" ]]; then
        echo "  Ya existe $AE_CKPT, se omite entrenamiento."
    else
        "$PY" - <<PYEOF
from eeggan.autoencoder_training_main import main
main([
    "data=$COND_TRAIN_CSV",
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

    echo "  -- GAN (AE-coupled, no-condicional) --"
    if [[ -f "$GAN_CKPT" ]]; then
        echo "  Ya existe $GAN_CKPT, se omite entrenamiento."
    else
        "$PY" - <<PYEOF
from eeggan.gan_training_main import main
main([
    "data=$COND_TRAIN_CSV",
    "kw_channel=Electrode",
    "kw_time=Time",
    "patch_size=$PATCH_SIZE",
    "n_epochs=$N_EPOCHS_GAN",
    "seed=$SEED",
    "autoencoder=$AE_CKPT",
    "save_name=$GAN_NAME",
])
PYEOF
    fi

    N_SAMPLES=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TEST_CSV'); print(df[df.Condition==$COND_VAL]['Trial'].nunique())")
    echo "  -- Generando $N_SAMPLES muestras (tantas como trials reales de test en esta condición) --"
    "$PY" - <<PYEOF
from eeggan.generate_samples_main import main
main([
    "model=$GAN_CKPT",
    "save_name=_tmp_${COND_NAME}.csv",
    "num_samples_total=$N_SAMPLES",
    "num_samples_parallel=$N_SAMPLES",
    "sequence_length=-1",
])
PYEOF
    # generate_samples no escribe columna Condition en modelos no-condicionales
    # -- se la pegamos a mano para que compare_subject()/Dataloader la vean.
    "$PY" -c "
import pandas as pd
df = pd.read_csv('generated_samples/_tmp_${COND_NAME}.csv')
df.insert(0, 'Condition', $COND_VAL)
df.to_csv('generated_samples/_tmp_${COND_NAME}.csv', index=False)
"
    rm -f "$COND_TRAIN_CSV"
done

OUT_CSV="${REPO_ROOT}/generated_samples/EEG_GAN_vanilla_s${SUBJECT_FMT}_synthetic.csv"
mkdir -p "${REPO_ROOT}/generated_samples"
"$PY" -c "
import pandas as pd
df = pd.concat([pd.read_csv('generated_samples/_tmp_NonTarget.csv'), pd.read_csv('generated_samples/_tmp_Target.csv')], ignore_index=True)
df.to_csv('$OUT_CSV', index=False)
print(f'  Guardado: $OUT_CSV ({len(df)} filas)')
"
rm -f generated_samples/_tmp_NonTarget.csv generated_samples/_tmp_Target.csv

echo ""
echo "=== Listo ==="
echo "Para agregarlo a la tabla de ablation (desde ttsgan-direct):"
echo "  cd $REPO_ROOT"
echo "  python eval_external_config.py --name eeg_gan_vanilla --gen-csv generated_samples/EEG_GAN_vanilla_s${SUBJECT_FMT}_synthetic.csv --subject $SUBJECT"
