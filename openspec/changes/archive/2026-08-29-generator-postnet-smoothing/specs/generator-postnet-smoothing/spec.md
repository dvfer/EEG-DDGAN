## ADDED Requirements

### Requirement: Optional residual smoothing of generator output
`TTSGenerator` SHALL support an optional PostNet-style residual 1D-convolution stack applied to its raw output, controlled by a boolean `use_postnet` flag passed at construction. When disabled, the generator's forward pass SHALL be byte-for-byte identical to the architecture without this change.

#### Scenario: PostNet disabled (default)
- **WHEN** `TTSGenerator` is constructed with `use_postnet=False` (or the argument omitted)
- **THEN** no PostNet submodule is instantiated and `forward()` returns exactly the base architecture's output, unchanged from before this change

#### Scenario: PostNet enabled
- **WHEN** `TTSGenerator` is constructed with `use_postnet=True`
- **THEN** a residual conv stack is instantiated and its output is added to the base generator's raw output before `forward()` returns, preserving the output tensor's shape `(batch, seq_len, channels)`

### Requirement: End-to-end training, no separate coupling
The PostNet module SHALL be trained jointly with the rest of the generator using the existing adversarial and feature-matching losses. It SHALL NOT require a separate pretraining stage, a frozen auxiliary network, or an additional encode/decode step to feed any discriminator.

#### Scenario: Training with PostNet enabled
- **WHEN** a GAN is trained with `use_postnet=True`
- **THEN** gradients from the generator loss (adversarial + feature-matching, including through `MultiscaleDWTDiscriminator` when active) flow into the PostNet parameters via the same backward pass as the rest of the generator, with no extra forward/backward pass through any other network

### Requirement: CLI-parametrizable
`use_postnet` SHALL be exposed as a CLI argument for `gan_training` following the existing `key=value` / bare-flag parsing convention, defaulting to `False`.

#### Scenario: Default training run
- **WHEN** `eeggan gan_training ...` is invoked without `use_postnet`
- **THEN** the GAN trains with `use_postnet=False`

#### Scenario: Opt-in training run
- **WHEN** `eeggan gan_training ... use_postnet` (or `use_postnet=True`) is invoked
- **THEN** the GAN trains with the PostNet smoothing stage enabled

### Requirement: moabb_pipeline.py config exposure
`moabb_pipeline.py` SHALL expose a `GAN_USE_POSTNET` config variable in its GAN configuration block, threaded through `train_gan()` into the `eeggan gan_training` CLI call.

#### Scenario: Pipeline run with PostNet toggled
- **WHEN** `GAN_USE_POSTNET = True` is set in `moabb_pipeline.py` and the pipeline is run
- **THEN** every subject's `eeggan gan_training` invocation includes `use_postnet` (or `use_postnet=True`)

### Requirement: Checkpoint self-describes PostNet presence
A trained checkpoint's saved configuration SHALL record whether its generator was built with `use_postnet=True`, so that reconstructing the generator architecture at inference time does not depend on the caller passing the correct flag.

#### Scenario: Loading a PostNet checkpoint for generation
- **WHEN** `generate_samples_main.py` loads a checkpoint whose saved configuration has `use_postnet=True`
- **THEN** it reconstructs `TTSGenerator` with `use_postnet=True` before loading the saved `state_dict`, without requiring the user to pass `use_postnet` on the `generate_samples` command line

#### Scenario: Loading a pre-existing checkpoint without the field
- **WHEN** `generate_samples_main.py` loads a checkpoint saved before this change (no `use_postnet` key in its configuration)
- **THEN** it treats `use_postnet` as `False` and reconstructs the generator without the PostNet stage, matching how that checkpoint was originally trained
