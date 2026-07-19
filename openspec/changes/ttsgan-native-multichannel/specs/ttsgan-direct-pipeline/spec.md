## ADDED Requirements

### Requirement: Autoencoder-free GAN construction
`init_gan` SHALL construct the generator and (primary) discriminator directly from `TTSGenerator`/`TTSDiscriminator`, with no code path that wraps them in `DecoderGenerator`/`EncoderDiscriminator` around an `Autoencoder`. The `autoencoder` CLI argument SHALL NOT be accepted by the GAN training command on this pipeline.

#### Scenario: GAN training without an autoencoder argument
- **WHEN** `gan_training` is run with a dataset and no `autoencoder` keyword
- **THEN** `init_gan` returns a plain `TTSGenerator` and `TTSDiscriminator` pair (optionally wrapped by the dual-discriminator setup), with no `Autoencoder`/`DecoderGenerator`/`EncoderDiscriminator` instances involved

#### Scenario: Autoencoder keyword is rejected
- **WHEN** `gan_training` is invoked with an `autoencoder=<path>` keyword
- **THEN** argument parsing SHALL reject it as an unrecognized keyword, matching the existing "keyword not recognized" behavior for any other unsupported argument

### Requirement: WGAN-GP training mechanics preserved
The training loop SHALL retain, unchanged in behavior for the non-autoencoder path, EEG-DDGAN's existing critic-iteration schedule, `WassersteinGradientPenaltyLoss` computation, checkpoint/history bookkeeping fields, and the DDP training entrypoint (`GANDDPTrainer`/`run`).

#### Scenario: Single-channel regression check
- **WHEN** a training run is executed with the same single-channel dataset and hyperparameters used before this change (no autoencoder, dual discriminator disabled)
- **THEN** the resulting discriminator/generator loss trajectory and saved checkpoint structure match pre-change EEG-DDGAN behavior within normal run-to-run variance

### Requirement: Dual wavelet discriminator retained
`MultiscaleDWTDiscriminator` and `StackingDiscriminator`, and their controlling CLI flags (`use_multiscale_dwt_discriminator`, `multiscale_dwt_high_freq`, `use_stacking`), SHALL remain fully functional on the autoencoder-free pipeline, producing the same `(validity, features)` output contract as before this change.

#### Scenario: Training with dual discriminator and stacking enabled
- **WHEN** `gan_training` is run with `use_multiscale_dwt_discriminator` and `use_stacking` both set, and no `autoencoder` keyword
- **THEN** training completes each batch by computing losses against both the primary and secondary discriminator's combined (stacked) output, without requiring any autoencoder-related configuration

### Requirement: Feature-matching loss retained
The L1 feature-matching loss computed between real and fake feature tensors — both the secondary (`MultiscaleDWTDiscriminator`) feature-matching term in `GANTrainer.batch_train`'s non-stacking branch and the stacking-branch feature-matching term computed via `StackingDiscriminator`'s combined features — SHALL remain in the generator loss computation, weighted by `lambda_fm=20`, unchanged in formula from EEG-DDGAN's current `trainer.py`, regardless of whether the autoencoder path exists or `n_channels` is `1` or `>1`.

#### Scenario: Feature matching contributes to generator loss with dual discriminator enabled
- **WHEN** a generator training step runs with `use_multiscale_dwt_discriminator` (and optionally `use_stacking`) enabled
- **THEN** the generator loss for that step includes the L1 feature-matching term between the secondary/stacked discriminator's real-data features and fake-data features, in addition to the adversarial validity term

#### Scenario: Feature matching disabled has no effect on generator loss beyond the primary discriminator
- **WHEN** a generator training step runs with `use_multiscale_dwt_discriminator` and `use_stacking` both disabled
- **THEN** the generator loss for that step SHALL NOT include any secondary-discriminator feature-matching term, matching current EEG-DDGAN behavior for the single-discriminator case

### Requirement: Checkpoint compatibility scoped to this pipeline
Checkpoints saved by the autoencoder-free pipeline SHALL be loadable for further training or inference only within this pipeline. Attempting to load a checkpoint that was trained with an autoencoder-wrapped generator/discriminator SHALL fail with a clear error rather than silently loading with mismatched shapes.

#### Scenario: Loading an incompatible autoencoder-era checkpoint
- **WHEN** a checkpoint whose `configuration['autoencoder']` is non-empty (i.e., trained with an autoencoder-wrapped model) is passed to this pipeline's training or generation entrypoint
- **THEN** loading SHALL fail with an explicit error (e.g., missing key, shape mismatch, or an explicit compatibility check) rather than producing incorrect samples
