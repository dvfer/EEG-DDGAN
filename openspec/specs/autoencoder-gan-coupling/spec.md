# autoencoder-gan-coupling

## Purpose

TBD — captures behavior of the frozen-autoencoder-wrapped GAN coupling (`DecoderGenerator`/`EncoderDiscriminator` around a `TransformerAutoencoder`/`TransformerDoubleAutoencoder`), including how condition columns interact with fixed-width AE sub-networks.

## Requirements

### Requirement: Condition columns SHALL NOT reach a fixed-width AE channel sub-network
When `EncoderDiscriminator` wraps a frozen autoencoder (single-level `channels`/`time` target, or two-level `full` target) and `n_conditions > 0`, the condition columns concatenated onto the discriminator's input on the channel axis SHALL be split off before the data reaches any AE sub-network whose channel-facing Linear layer width was fixed at AE-training time, and reattached to the encoded output before it reaches the discriminator.

#### Scenario: Conditional target=channels discriminator input
- **WHEN** `EncoderDiscriminator.forward` receives real or fake data of shape `(batch, seq, channels_in + n_conditions)` wrapping a `channels`-target `TransformerAutoencoder`
- **THEN** the AE's `encode()` is called only on the `channels_in`-wide slice, and the `n_conditions`-wide slice is reattached to the encoded result before it is passed to the underlying discriminator

#### Scenario: Conditional target=full discriminator input
- **WHEN** `EncoderDiscriminator.forward` receives real or fake data of shape `(batch, seq, channels_in + n_conditions)` wrapping a `full`-target `TransformerDoubleAutoencoder`
- **THEN** the AE's `encode()` is called only on the `channels_in`-wide slice, and the `n_conditions`-wide slice is reattached to the encoded result before it is passed to the underlying discriminator
- **AND** no `RuntimeError` (shape/width mismatch) is raised regardless of `n_conditions`

#### Scenario: Unconditional discriminator input is unaffected
- **WHEN** `EncoderDiscriminator.forward` receives data with `n_conditions == 0`
- **THEN** behavior is unchanged from calling `self.encoder.encode(data)` directly with no split/reattach

### Requirement: target=full is a supported, untested-no-longer AE-GAN coupling target
`train_eeggan_vanilla.sh` SHALL support `target=full` as a working option on equal footing with `time` and `channels`, with no early-exit guard, once condition columns are correctly isolated from the double autoencoder's channel sub-network.

#### Scenario: Running the vanilla script with target=full
- **WHEN** a user runs `train_eeggan_vanilla.sh <subject> full`
- **THEN** the script does not exit early with a "no funciona" error, and proceeds through autoencoder training, GAN training, and sample generation
