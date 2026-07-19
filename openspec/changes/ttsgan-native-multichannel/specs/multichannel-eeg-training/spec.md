## ADDED Requirements

### Requirement: Dataloader produces validated multi-channel tensors
`Dataloader`, when constructed with `kw_channel` pointing to a column with `N > 1` distinct electrode values, SHALL reshape the loaded data into `(trial, sequence, channel)` form with the channel dimension equal to `N`, and expose `self.channels` with length `N`, without requiring single-channel-specific handling downstream.

#### Scenario: Loading a multi-electrode dataset
- **WHEN** a CSV dataset with `N > 1` distinct values in the `kw_channel` column is loaded via `Dataloader`
- **THEN** `get_data()` returns a tensor whose channel dimension equals `N`, and `self.channels` contains `N` distinct channel identifiers

### Requirement: Generator and primary discriminator channel dimensions match `n_channels` end-to-end
For any `n_channels > 1` derived from the dataloader, `init_gan` SHALL configure `TTSGenerator` to output `n_channels` channels and `TTSDiscriminator` to accept `n_channels + n_conditions` input channels, with no hardcoded assumption of `n_channels == 1` remaining in the non-autoencoder path.

#### Scenario: Initializing the GAN for multichannel data
- **WHEN** `init_gan` is called with `n_channels = N` for `N > 1` (no autoencoder)
- **THEN** the returned generator's output channel dimension equals `N`, and the returned discriminator accepts inputs with `N + n_conditions` channels without a shape error

### Requirement: `MultiscaleDWTDiscriminator` sizes its per-scale heads for `n_channels > 1`
`MultiscaleDWTDiscriminator` SHALL correctly size and run its per-scale low/high-frequency MLP heads when constructed with `in_channels = N` for `N > 1`, producing a validity score and a feature tensor whose dimensions are consistent with `N`.

#### Scenario: Forward pass with multichannel input
- **WHEN** `MultiscaleDWTDiscriminator(in_channels=N, ...)` for `N > 1` receives a batch of shape `(B, seq_len, N)` (or `(B, N, seq_len)`)
- **THEN** the forward pass completes without a shape-mismatch error and returns `(validity, combined_features)` with `combined_features` sized consistently with `N` and the configured scale/head parameters

### Requirement: WGAN-GP gradient penalty handles multi-channel inputs
`WassersteinGradientPenaltyLoss._gradient_penalty` SHALL correctly compute the gradient penalty for real/fake tensors whose channel dimension (including concatenated conditions) is `N > 1`, producing a finite penalty value.

#### Scenario: Gradient penalty with multichannel real/fake batches
- **WHEN** `_gradient_penalty` is invoked with real and fake tensors of matching shape where the channel dimension is `N + n_conditions` for `N > 1`
- **THEN** the interpolation, discriminator forward pass, and gradient computation succeed and return a finite, non-NaN penalty value

### Requirement: Feature-matching loss remains correct for `n_channels > 1`
The L1 feature-matching loss between real and fake feature tensors (secondary-discriminator and stacking branches, per the `ttsgan-direct-pipeline` feature-matching requirement) SHALL compute correctly when `n_channels = N > 1`, i.e. the real-data and fake-data feature tensors compared by the L1 loss SHALL have matching shapes derived consistently from the same `N`-channel `MultiscaleDWTDiscriminator`/`StackingDiscriminator` configuration.

#### Scenario: Feature matching with multichannel input
- **WHEN** a generator training step runs with `n_channels = N > 1` and `use_multiscale_dwt_discriminator` (optionally `use_stacking`) enabled
- **THEN** the real-data and fake-data feature tensors passed to the L1 feature-matching loss have identical shape, and the resulting feature-matching term is a finite scalar included in the generator loss

### Requirement: End-to-end multichannel training is validated on real data
The autoencoder-free, dual-discriminator pipeline SHALL be exercised through a short training run on a real multi-electrode EEG dataset (`n_channels > 1`), confirming it completes without shape or dtype errors and produces a checkpoint whose generated-sample format matches the existing single-channel checkpoint convention (condition columns + channel label + time series, stacked per channel).

#### Scenario: Short multichannel training run
- **WHEN** `gan_training` is run for a small number of epochs on a real dataset with `N > 1` electrodes, with `use_multiscale_dwt_discriminator` and `use_stacking` enabled and no `autoencoder` keyword
- **THEN** training completes all epochs without a shape or dtype error, and the saved checkpoint's `samples` entries have the same column structure (conditions, channel label, time-series values) as an equivalent single-channel run, generalized to `N` channels
