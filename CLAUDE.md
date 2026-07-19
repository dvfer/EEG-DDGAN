# CLAUDE.md

Reference guide for AI assistants (and humans) working in this repository.

---

## 1. Project Overview

**EEG-DDGAN** is a PyTorch package that uses **Generative Adversarial Networks (GANs)** to synthesize trial-level EEG data. The synthetic samples can augment real data to improve downstream classifiers (SVM, neural nets, etc.).

- **Upstream:** `AutoResearch/EEG-GAN` (Brown University, Autonomous Empirical Research Group). Canonical docs: https://autoresearch.github.io/EEG-GAN/
- **This fork:** `dvfer/EEG-DDGAN` — customized for **P300 generation** from the MOABB `BNCI2014_009` dataset, via the custom driver `moabb_pipeline.py`.
- **Core method:** Wasserstein GAN with Gradient Penalty (WGAN-GP) using a patch-based Transformer generator/discriminator (the "TTS-GAN" family), optionally coupled with a multi-scale Discrete-Wavelet-Transform (DWT) discriminator (+ a stacking meta-discriminator combining both).
- **Branch note (`ttsgan-direct`):** this branch trains **directly on raw (optionally multi-channel) EEG** — the autoencoder-coupled generator/discriminator path (`DecoderGenerator`/`EncoderDiscriminator` around a `TransformerAutoencoder`) was removed from `init_gan`/`gan_training_main.py`/`system_inputs.py`. It remains available on `main`. See `openspec/changes/ttsgan-native-multichannel/`.
- **Version:** 2.0.2 (see `pyproject.toml`).

> Note: The `mkdocs.yml` at the repo root is **stale** — the doc sources it references were deleted in commit `3548eb9`. Only `README.md`, `LICENSE.md`, and `eeggan/trained_models/pretrained_gan.pt` remain as non-code artifacts.

---

## 2. Tech Stack & Dependencies

- **Language:** Python >= 3.7
- **Deep learning:** PyTorch 2.3.x (`torch`, `torchvision`, `torchaudio`), `torchsummary`, `einops`
- **Data/science:** `pandas` 2.2.x, `numpy` <2.0, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`
- **Build system:** `hatchling` (declared in `pyproject.toml`)

### Undeclared / implicit dependencies (not in `pyproject.toml`)
- **`pytorch_wavelets`** — required by `MultiscaleDWTDiscriminator` (`eeggan/nn_architecture/models.py:102`). Install manually if you enable `use_multiscale_dwt_discriminator`.
- **`requests`** — used only by `setup_tutorial_main.py` to download example data.
- **`moabb`** (+ `mne`) — required **only** by the fork-specific `moabb_pipeline.py`, not by the `eeggan` package itself.

---

## 3. Repository Layout

```
EEG-DDGAN/
├── moabb_pipeline.py          # [FORK] End-to-end MOABB→CSV→AE→GAN driver (P300/BNCI2014_009)
├── pyproject.toml             # Build/dependency spec; registers the `eeggan` console script
├── mkdocs.yml                 # STALE (doc sources were deleted upstream)
├── README.md / LICENSE.md
└── eeggan/                    # The Python package
    ├── __main__.py            # CLI router: dispatches `eeggan <command>`
    ├── gan_training_main.py   # GAN training entry point  → trained_models/
    ├── autoencoder_training_main.py  # AE training        → trained_ae/
    ├── vae_training_main.py   # VAE training (alternative)→ trained_vae/
    ├── generate_samples_main.py # Inference              → generated_samples/
    ├── visualize_main.py      # Plotting / PCA / t-SNE / FFT / spectrogram
    ├── get_gan_config.py      # Print a checkpoint's configuration dict
    ├── setup_tutorial_main.py # Download example data + pretrained models
    ├── helpers/               # Core logic
    │   ├── dataloader.py        # `Dataloader`: CSV → (trial, seq, channel) tensors
    │   ├── trainer.py           # `GANTrainer`, `AETrainer`, `VAETrainer` (~1087 lines)
    │   ├── ddp_training.py      # DDP-wrapped trainers + `mp.spawn` orchestration
    │   ├── get_master.py        # `find_free_port()` for DDP
    │   ├── initialize_gan.py    # `init_gan()` factory: builds G/D (+ DWT + stacking)
    │   ├── system_inputs.py     # CUSTOM CLI parser + per-command arg schemas
    │   ├── moabb_export.py      # MOABB/MNE Epochs → long-format CSV
    │   ├── visualize_pca.py     # PCA/t-SNE (adapted from TimeGAN, NeurIPS 2019)
    │   └── visualize_spectogram.py  # FFT histogram + spectrogram plots
    ├── nn_architecture/       # Neural network definitions
    │   ├── models.py            # GAN zoo: TTS G/D, AE-wrapped G/D, DWT disc, stacking
    │   ├── tts_gan_components.py# Low-level TTS-GAN blocks (from imics-lab/tts-gan)
    │   ├── ae_networks.py       # `Autoencoder`, `TransformerAutoencoder`, `TransformerDoubleAutoencoder`
    │   ├── vae_networks.py      # `VariationalAutoencoder`
    │   └── losses.py            # `WassersteinGradientPenaltyLoss` (the one actually used)
    ├── auxiliary/             # Standalone helper scripts (run directly)
    │   ├── checkpoint_to_csv.py # Extract samples/losses from a .pt to CSV
    │   ├── create_averaged_erps.py  # Avg trials→ERPs (WARNING: hardcoded Windows paths)
    │   └── data_downsampling.py # Linear-interpolation downsampling
    └── trained_models/
        └── pretrained_gan.pt  # 12 MB tutorial checkpoint (only pretrained artifact present)
```

---

## 4. The Core Workflow

The pipeline is driven by the `eeggan <command>` CLI (see `eeggan/__main__.py:11`).

```
   ┌──────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
   │  (optional) AE       │   │  GAN training        │   │  generate_samples  │
   │  autoencoder_training├──▶│  gan_training        ├──▶│  generate_samples  ├──▶ synthetic CSV
   │  → trained_ae/*.pt   │   │  → trained_models/*  │   │  → generated_samples/
   └──────────────────────┘   └──────────────────────┘   └────────────────────┘
                                                                       │
                                                                       ▼
                                                              visualize (loss/avg/PCA/tSNE/FFT)
```

1. **Autoencoder** (optional, `main` branch only — not on `ttsgan-direct`) — compresses channels/time into a latent space. Saves to `trained_ae/`.
2. **GAN training** — TTS generator + discriminator (on `ttsgan-direct`: always trained directly on raw data, never AE-wrapped), with an optional DWT secondary discriminator (+ optional stacking meta-discriminator). Saves to `trained_models/`.
3. **Generate samples** — load a `.pt` checkpoint, draw latent vars conditioned on labels, write synthetic trial CSVs to `generated_samples/`.
4. **Visualize** — inspect losses, averaged curves, PCA/t-SNE vs. real data, FFT/spectrograms.

`setup_tutorial` bootstraps example data + models (creates `data/`, `trained_ae/`, `trained_models/`, `generated_samples/`, `trained_vae/` and downloads from the upstream `dev` branch).

---

## 5. CLI Usage

The `eeggan` console script (`eeggan.__main__:main`) recognizes these commands:
`gan_training`, `autoencoder_training`, `vae_training`, `generate_samples`, `visualize`, `setup_tutorial`, `help`.

### Custom argument syntax (NOT argparse)
The parser lives in `eeggan/helpers/system_inputs.py:415` (`parse_arguments`). Each command declares a schema via `default_inputs_*()` (e.g. `system_inputs.py:258`). Syntax:

| Form | Example | Meaning |
|---|---|---|
| `key=value` (no spaces around `=`) | `data=foo.csv`, `n_epochs=2000` | Value coerced to the declared type |
| Bare keyword (flag) | `use_multiscale_dwt_discriminator`, `ddp` | Sets that boolean key to `True` |
| Comma list | `kw_conditions=A,B` or `conditions=1,0` | Parsed as a list (auto-typed per element) |
| `help` | `eeggan gan_training help` | Prints the parameter table and exits |

### Examples

```bash
# One-time: fetch example data + pretrained models
eeggan setup_tutorial

# Train an autoencoder (multichannel: target=full trains two stacked levels)
eeggan autoencoder_training data=data/subject_001.csv \
    kw_channel=Electrode kw_time=Time \
    target=full channels_out=10 time_out=10 \
    n_epochs=2000 seed=42 save_name=AE_s001

# Train a GAN directly on raw (optionally multichannel) data
# (multiscale DWT discriminator + high-freq coefficients + stacking meta-discriminator)
# NOTE: 'autoencoder=' is not accepted on the ttsgan-direct branch (main only).
eeggan gan_training data=data/subject_001.csv \
    kw_channel=Electrode kw_conditions=Condition kw_time=Time \
    patch_size=10 n_epochs=2000 seed=42 \
    use_multiscale_dwt_discriminator multiscale_dwt_high_freq=True use_stacking \
    save_name=GAN_s001

# Generate synthetic samples (conditions: -1 = random for binary)
eeggan generate_samples model=trained_models/GAN_s001.pt \
    num_samples_total=1000 num_samples_parallel=50 \
    conditions=1 sequence_length=-1 save_name=gen_s001

# Visualize
eeggan visualize model=trained_models/checkpoint.pt loss average pca
eeggan visualize data=data.csv kw_conditions=Condition kw_time=Time n_samples=20

# Inspect a checkpoint's config
eeggan get_gan_config model=trained_models/pretrained_gan.pt
```

You can also call the `*_main.main(args)` functions **directly** from Python by passing a list of these token strings — this is exactly what `moabb_pipeline.py` does.

---

## 6. Input Data Format

The `Dataloader` (`eeggan/helpers/dataloader.py:11`) expects a single CSV in **long format**:

- **Each row = one (trial × channel) combination.** For N trials × C channels there are N×C rows.
- **Time-series values are spread across many wide columns** (one per time step).
- Multichannel: the same trial is repeated on C consecutive rows, each tagged with a different channel/electrode.

Example layout:

| ParticipantID | Condition | Trial | Electrode | Time1 | Time2 | ... | TimeN |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | Fz | ... | ... | ... | ... |
| 1 | 1 | 0 | Cz | ... | ... | ... | ... |
| 1 | 0 | 1 | Fz | ... | ... | ... | ... |

### The three keyword arguments
- **`kw_time`** (default `'Time'`) — **substring match**. Every column whose name *contains* this substring is treated as a time-series data column.
- **`kw_conditions`** (default `''`) — a column name, or comma-separated list of names, holding integer/binary class labels. Empty string = no conditions.
- **`kw_channel`** (default `None`) — column identifying the channel/electrode. When set, the loader reshapes data to `(trial, sequence, channel)`. When unset, treated as single-channel.

### Transformations (applied in order; inverse params stored)
`diff_data` (first difference) → `norm_data` (min-max to [0,1]) → `std_data` (z-score). **Note:** `gan_training_main.py` *hardcodes* `norm_data=True, std_data=False, diff_data=False` (`gan_training_main.py:59-61`); these are not user-overridable via CLI for GAN training.

Final tensor shape: **(n_trials, n_conditions + sequence_length, n_channels)** — labels are concatenated onto the sequence dimension.

Helpers to produce this format: `eeggan/helpers/moabb_export.py:26` (`export_to_csv`) and `moabb_pipeline.py:81` (fork version).

---

## 7. Architecture Notes

### Loss: Wasserstein GAN + Gradient Penalty (WGAN-GP)
- The only loss wired into `GANTrainer`: `WassersteinGradientPenaltyLoss` (`eeggan/nn_architecture/losses.py:61`). Gradient penalty via `autograd.grad` (`losses.py:73`).
- Defaults: **critic updated 5× per generator step** (`critic_iterations`, `trainer.py:54`), **λ_gp = 10** (`trainer.py:55`), Adam betas `(0.0, 0.9)`, latent_dim = 128 (`gan_training_main.py:92`).
- **Feature matching:** generator loss adds an L1 feature-matching term, **λ_fm = 20** (`trainer.py:351,385`), between real and fake penultimate features.

### Generator / Discriminator (TTS-GAN, patch-based)
- Patch-based ViT-style transformer blocks adapted from `imics-lab/tts-gan` (`tts_gan_components.py`).
- The discriminator's `ClassificationHead` returns **both** validity and penultimate features → enables feature matching.
- **Patch divisibility constraint:** the (possibly AE-encoded) sequence length must be divisible by `patch_size` (`gan_training_main.py:152`). If using an AE with `target='time'`/`'full'`, `time_out` must satisfy this.

### Conditional generation
Conditions (`kw_conditions`) are appended to the generator's latent input (`latent_dim + n_conditions`, `gan_training_main.py:155`) and to the discriminator's channel input (`gan_training_main.py:156`). At inference, `conditions=-1`/`-2` selects randomly.

### Autoencoder–GAN coupling (`main` branch only)
On `main`, a frozen `TransformerDoubleAutoencoder` can wrap the GAN: `DecoderGenerator` decodes generator output to raw space; `EncoderDiscriminator` encodes real/fake into latent space before discrimination (`models.py:36,68`). This lets the GAN learn in a compressed representation. **On `ttsgan-direct`, this path was removed from `init_gan`** — the GAN always trains directly on raw (optionally multi-channel) data; `autoencoder=` is not an accepted CLI argument here. See `openspec/changes/ttsgan-native-multichannel/`.

### Multi-scale DWT discriminator
`MultiscaleDWTDiscriminator` (`models.py:104`) uses `pytorch_wavelets.DWT1DForward` with **J=4 progressive decomposition levels** and **`db4`** wavelet (`models.py:110,114`). Each level has MLP heads for low-freq approximation coefficients and (optionally, via `multiscale_dwt_high_freq`) high-freq detail coefficients. Activated by `use_multiscale_dwt_discriminator`. **Requires the undeclared `pytorch_wavelets` package.**

### Stacking discriminator (meta-learner)
`StackingDiscriminator` (`models.py:219`) concatenates primary (TTS) + secondary (DWT) features through a `meta_head`. Enabled by `use_stacking`; bypasses to primary when secondary is unavailable.

> **Important — discriminator flag caveat:** The CLI declares `use_dwt_discriminator`, `use_scattering_discriminator`, and `use_spectrogram_discriminator`, but `init_gan()` (`initialize_gan.py:18`) **only wires up `use_multiscale_dwt_discriminator`**. The others are no-ops.

### Other implementation details
- **DDP:** `ddp_training.py` + `get_master.find_free_port()` + `mp.spawn` (`run` at `ddp_training.py:122`). nccl backend; each GPU trains on the whole dataset (effective epochs × world_size).
- **Attention backend pinning:** `gan_training_main.py` disables flash/mem-efficient SDPA and forces `SDPBackend.MATH` for stability.
- **Two-phase checkpointing:** toggles between `checkpoint_01.pt`/`checkpoint_02.pt` every `sample_interval` epochs; consolidated to `checkpoint.pt` on completion.

---

## 8. `moabb_pipeline.py` (fork-specific)

The main automation artifact added by this fork (`moabb_pipeline.py:220`, `main()`). It replaces bash training loops and calls the eeggan `main()` functions directly. Configure the variables at the top of the file:

```python
SUBJECTS = [1, 2, 3]        # BNCI2014_009 subject IDs (1-10)
CONDITION = 'Both'          # 'Target', 'NonTarget', or 'Both'
NORM = False
DATA_DIR = 'subject_data/train';  AE_DIR = 'trained_ae';  GAN_DIR = 'trained_models'
MODEL_PREFIX = 'GAN_009_Modded'
# Autoencoder: target='time', time_out=50, n_epochs=2000
# GAN: patch_size=10, n_epochs=2000, use_multiscale_dwt_discriminator, high_freq=True
```

Per subject it:
1. Loads MOABB `BNCI2014_009` P300 data → `X` of shape `(n_trials, n_channels, n_timepoints)`.
2. Writes the long-format CSV to `subject_data/train/subject_XXX.csv` (`export_to_csv`, `moabb_pipeline.py:81`).
3. Trains the GAN directly on the raw multi-channel CSV (no autoencoder step on this branch) → `trained_models/GAN_009_Modded_sXXX.pt`.

`patch_size` must divide the dataset's raw sequence length (there is no AE `time_out` to size against anymore); `gan_training_main.py` raises a clear `ValueError` if it doesn't.

Requires the `moabb` (+ `mne`) packages. Run with `python moabb_pipeline.py`.

---

## 9. Key Files Reference

| Concept | Location |
|---|---|
| CLI router | `eeggan/__main__.py:11` |
| Custom arg parser | `eeggan/helpers/system_inputs.py:415` (`parse_arguments`) |
| GAN arg schema | `eeggan/helpers/system_inputs.py:258` (`default_inputs_training_gan`) |
| AE arg schema | `eeggan/helpers/system_inputs.py:289` |
| GAN training entry | `eeggan/gan_training_main.py` (`main`) |
| GAN factory (builds G + D) | `eeggan/helpers/initialize_gan.py:18` (`init_gan`) |
| WGAN-GP loss | `eeggan/nn_architecture/losses.py:61` (penalty at `:73`) |
| GANTrainer (loop) | `eeggan/helpers/trainer.py:40` |
| Latent sampling | `eeggan/helpers/trainer.py:652` (`sample_latent_variable`) |
| Dataloader (CSV) | `eeggan/helpers/dataloader.py:11` |
| MOABB → CSV export | `eeggan/helpers/moabb_export.py:26` |
| TTS Generator/Discriminator | `eeggan/nn_architecture/models.py:24,31` |
| AE-wrapped G/D | `eeggan/nn_architecture/models.py:36,68` |
| DWT discriminator | `eeggan/nn_architecture/models.py:104` |
| Stacking discriminator | `eeggan/nn_architecture/models.py:219` |
| Transformer autoencoders | `eeggan/nn_architecture/ae_networks.py:88,139` |
| VAE | `eeggan/nn_architecture/vae_networks.py:9` |
| AETrainer / VAETrainer | `eeggan/helpers/trainer.py:679,896` |
| DDP orchestration | `eeggan/helpers/ddp_training.py:122` (`run`) |
| Fork pipeline driver | `moabb_pipeline.py:220` (`main`) |

---

## 10. Development Notes

### Install
```bash
pip install -e .          # editable install; registers the `eeggan` console script
pip install pytorch_wavelets   # ONLY if using the DWT discriminator
```

### Testing
- **There is no test suite, test framework config, or CI.** No `tests/`, `pytest.ini`, `conftest.py`, or `.github/` exist. Verify changes by running the relevant `eeggan <command>` end-to-end on a small CSV.

### Common pitfalls
- **`patch_size` divisibility:** the effective sequence length must be divisible by `patch_size` (`gan_training_main.py:152`). With an AE using `target='time'`/`'full'`, this applies to `time_out` instead.
- **`norm_data` hardcoded:** GAN training always min-max normalizes to [0,1] (`gan_training_main.py:61`); `std_data`/`diff_data` flags are ignored for GAN training.
- **Undeclared DWT dependency:** `pytorch_wavelets` is imported but not in `pyproject.toml`.
- **Stale docs:** `mkdocs.yml` references deleted files; canonical docs are upstream at https://autoresearch.github.io/EEG-GAN/
- **Hardcoded paths:** `eeggan/auxiliary/create_averaged_erps.py` contains Windows dev paths (`C:\Users\Daniel\...`) and is not runnable as-is on this system.
- **Pretrained weights:** only `eeggan/trained_models/pretrained_gan.pt` (12 MB) is present. Run `eeggan setup_tutorial` to fetch example CSVs and an AE checkpoint.

---

## 11. Git Context

- **Remote:** `git@github.com:dvfer/EEG-DDGAN.git` (origin).
- **Upstream origin of the code:** `AutoResearch/EEG-GAN`.
- **Branches:**
  - `main` — autoencoder-capable pipeline (the one described generically above unless a branch note says otherwise).
  - `mv-train` (remote) — added multivariate/MOABB support: the `MultiscaleDWTDiscriminator.forward` dim-based reshape fix and `moabb_pipeline.py`/`moabb_export.py`.
  - `ttsgan-direct` (local, based on `mv-train`) — this branch. Removes the autoencoder-coupled GAN path entirely (`init_gan`, `gan_training_main.py`, `system_inputs.py`, `generate_samples_main.py`), keeps the dual wavelet discriminator + feature matching + stacking, and adapts `moabb_pipeline.py` to train directly on raw multi-channel data. Tracked via `openspec/changes/ttsgan-native-multichannel/`.
- **Notable commits (pre-fork history):**
  - `3548eb9` — "delete docs, pretrained weights" (removed the `docs/` sources + most pretrained models; `mkdocs.yml` left behind and is now stale).
  - `cc99608` — "EEG-DDGAN" (the fork-specific additions, including `moabb_pipeline.py`).
