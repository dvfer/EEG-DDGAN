# Implementation notes

Freeform working notes for `ttsgan-native-multichannel`, not a tracked OpenSpec artifact.

## Branch base (superseding design.md Decision 1's literal wording)

`ttsgan-direct` was created from `origin/mv-train` (commit `a3a10e9`, "feat: soporte multivariado"), not from `main`, because `mv-train` already contains work directly relevant to this change:

- `eeggan/nn_architecture/models.py::MultiscaleDWTDiscriminator.forward` already replaces the ambiguous `if x.shape[-1] == self.in_channels: permute` heuristic with an explicit `x.dim() == 4` (gradient-penalty path) / `x.dim() == 3` (trainer path) branch — this is the exact fix design.md's Risk 3 and tasks 4.3/4.6 called for. Verified present; no further fix needed, only verification once channel dims actually vary.
- `eeggan/helpers/moabb_export.py` and `moabb_pipeline.py` (repo root) already implement a full MOABB → CSV → AE → GAN pipeline for the real multi-electrode `BNCI2014_009` P300 dataset — this satisfies task 5.1's "obtain or prepare a real multi-electrode EEG dataset" requirement, once adapted (see below).
- `pyproject.toml` on this branch already declares `moabb`, `mne`, `pytorch_wavelets`, `PyWavelets` as dependencies (not yet installed in the working environment as of this session).

**Consequence for scope**: `moabb_pipeline.py::train_gan` currently calls `gan_main` with `autoencoder={ae_save_path}` — i.e. it trains the GAN through the autoencoder-coupled path this change removes. It must be updated (not just left alone) to train without an autoencoder once section 2's removal lands, or the pipeline will break by passing a now-unrecognized `autoencoder=` keyword. Tracked as an added task under section 2/5.

## Task 1.2 — baseline config for the regression check

No local example dataset or non-autoencoder checkpoint exists in this repo (only `eeggan/trained_models/pretrained_gan.pt`, which was itself trained *with* an autoencoder — `configuration['autoencoder'] = 'trained_ae/pretrained_autoencoder.pt'` — so it is not directly usable as the "no autoencoder" baseline task 3 needs). Its hyperparameters are still useful as realistic single-channel defaults:

```
sequence_length=100, num_layers=4, hidden_dim=16, latent_dim=128, batch_size=128,
patch_size=10, n_conditions=1, n_channels=1, kw_channel=Electrode, kw_conditions=Condition,
kw_time=Time, data=data/eeggan_training_example.csv, epochs=2000
```

Recorded baseline command for the section-3 regression check (single-channel, no autoencoder, dual discriminator + stacking enabled, matching the style already used by `moabb_pipeline.py::train_gan`):

```bash
eeggan setup_tutorial   # fetches data/eeggan_training_example.csv if not already present

eeggan gan_training \
    data=data/eeggan_training_example.csv \
    kw_channel=Electrode kw_conditions=Condition kw_time=Time \
    patch_size=10 hidden_dim=16 num_layers=4 batch_size=128 \
    n_epochs=100 sample_interval=10 seed=42 \
    use_multiscale_dwt_discriminator multiscale_dwt_high_freq=True use_stacking \
    save_name=baseline_ttsgan_direct_singlechannel.pt
```

`n_epochs` reduced from the pretrained checkpoint's 2000 to 100 for a practical regression-check runtime; `sample_interval=10` so checkpoints/samples are captured periodically within that shorter run. Not yet executed — see task-status notes below.

## Environment status (this session)

- `torch` 2.9.1+cu128 is installed globally, but `moabb` is **not** installed (`ModuleNotFoundError: No module named 'moabb'`). `pytorch_wavelets` availability unconfirmed at time of writing.
- No dataset CSVs present under the repo; `data/eeggan_training_example.csv` needs `eeggan setup_tutorial` (network fetch from upstream EEG-GAN's `dev` branch) or equivalent.
- Sections 3 (regression run) and 5 (real multichannel training run) require installing `moabb`/`mne`/`pytorch_wavelets`, network access to fetch data, and non-trivial training time — these were left for the user to run/confirm rather than executed automatically in this session. All other tasks (code edits in sections 1, 2, 4, and the `moabb_pipeline.py` adaptation) were completed and are ready for these runs to validate.
