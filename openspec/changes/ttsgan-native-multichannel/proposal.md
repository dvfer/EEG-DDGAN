## Why

EEG-DDGAN currently inherits EEG-GAN's full architecture, including the optional autoencoder-wrapped generator/discriminator path (`DecoderGenerator`/`EncoderDiscriminator` around `TransformerAutoencoder`/`TransformerDoubleAutoencoder`). That layer exists in EEG-GAN to let the GAN operate on a compressed representation, but it adds a second training stage, extra checkpoint coupling, and shape-adjustment logic (`init_gan` recomputing `n_channels`/`sequence_length_generated` from the autoencoder's dimensions) that the thesis's dual-discriminator work does not need and that makes the codebase harder to reason about independently of EEG-GAN. At the same time, EEG-DDGAN's wavelet dual-discriminator setup (`MultiscaleDWTDiscriminator` + `StackingDiscriminator`) has only ever been trained and validated on single-channel EEG (`n_channels=1`), even though the long-term goal is multivariate EEG generation. Decoupling the generator/discriminator pipeline from EEG-GAN's autoencoder abstraction now — while the codebase is still small — makes it tractable to then validate and harden multi-channel support on top of a simpler, TTS-GAN-native base.

## What Changes

- Introduce a TTS-GAN-native training path in EEG-DDGAN that builds the generator/discriminator directly from `nn_architecture/tts_gan_components.py` (as EEG-GAN itself does when no autoencoder is given), without ever routing through `DecoderGenerator`/`EncoderDiscriminator`/`Autoencoder`.
- **BREAKING**: Remove the autoencoder-conditioned code path from `helpers/initialize_gan.py::init_gan` for this pipeline — the `autoencoder` CLI argument, `ae_networks.py` (`TransformerAutoencoder`, `TransformerDoubleAutoencoder`, `Autoencoder`), and the corresponding branch in `gan_training_main.py`/`autoencoder_training_main.py` are dropped from this branch's GAN training flow. `autoencoder_training_main.py`/`vae_training_main.py` CLI commands are no longer relevant to GAN training on this branch.
- Retain, unmodified in behavior, everything EEG-GAN's WGAN-GP trainer already does that is not autoencoder-specific: `GANTrainer`'s critic-iteration schedule, `WassersteinGradientPenaltyLoss`, checkpointing/history bookkeeping, DDP training (`ddp_training.py`), and the CLI argument system (`system_inputs.py`).
- Retain EEG-DDGAN's dual-discriminator additions as-is: `MultiscaleDWTDiscriminator`, `StackingDiscriminator`, the feature-matching loss, and the `use_multiscale_dwt_discriminator`/`multiscale_dwt_high_freq`/`use_stacking` CLI flags — these become the primary/secondary discriminator setup for the new pipeline instead of being layered on top of an autoencoder-wrapped model.
- Make the number of EEG channels (`n_channels`) a validated, first-class dimension through the whole pipeline — `Dataloader`'s multi-channel reshape path, generator output channels, primary (TTS) discriminator input channels, and `MultiscaleDWTDiscriminator`'s per-scale MLP head sizing (currently implicitly assumes `in_channels` consistent with a single-channel run) — and confirm training runs end-to-end with `n_channels > 1` on a real multi-electrode dataset, not just synthetic single-channel data.
- Work happens on a new git branch inside `EEG-DDGAN` (not upstream `EEG-GAN`); no changes are made to the `EEG-GAN` repo.

## Capabilities

### New Capabilities
- `ttsgan-direct-pipeline`: GAN training pipeline built directly on TTS-GAN generator/discriminator primitives (WGAN-GP, checkpointing, DDP) with no autoencoder stage, carrying forward the dual wavelet-discriminator/stacking setup from EEG-DDGAN.
- `multichannel-eeg-training`: End-to-end support and validation for training the `ttsgan-direct-pipeline` on multivariate (multi-electrode) EEG data, covering dataloader reshaping, generator/discriminator channel dimensions, and the wavelet discriminator's per-channel handling.

### Modified Capabilities
(none — this is the first OpenSpec change tracked for this repo)

## Impact

- **Affected code**: `eeggan/helpers/initialize_gan.py` (`init_gan`), `eeggan/gan_training_main.py`, `eeggan/helpers/system_inputs.py` (drop/gate `autoencoder` arg for this path), `eeggan/nn_architecture/models.py` (`MultiscaleDWTDiscriminator`, `StackingDiscriminator` channel handling), `eeggan/helpers/dataloader.py` (multi-channel reshape path), `eeggan/helpers/trainer.py` (`GANTrainer`/`GANDDPTrainer` — confirm no autoencoder-only assumptions leak in).
- **Not affected**: `EEG-GAN` upstream repo (read-only reference), `eeggan/nn_architecture/ae_networks.py` and VAE/autoencoder CLI commands stay available on `main` for anyone still using that path — they are just unused by this new branch's GAN training flow.
- **Dependencies**: no new third-party dependencies expected beyond the existing `pytorch_wavelets` already used by `MultiscaleDWTDiscriminator`.
- **Data**: requires a real multi-electrode EEG dataset (not the single-channel example data) to validate the multichannel capability.
