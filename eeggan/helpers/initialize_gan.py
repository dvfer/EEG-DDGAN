from eeggan.nn_architecture.models import TTSGenerator, TTSDiscriminator, MultiscaleDWTDiscriminator


gan_architectures = {
        'TTSGenerator': lambda seq_len, hidden_dim, patch_size, channels, latent_dim, num_layers, num_heads, use_postnet=False, **kwargs: TTSGenerator(seq_len, patch_size, channels, 1, latent_dim, 10, num_layers, num_heads, 0.5, 0.5, use_postnet=use_postnet),
        'TTSDiscriminator': lambda channels, hidden_dim, patch_size, seq_len, num_layers, **kwargs: TTSDiscriminator(channels, patch_size, 50, seq_len, num_layers, 1),
    }

gan_types = {
        'tts': ['TTSGenerator', 'TTSDiscriminator'],
    }


def init_gan(latent_dim_in,
             channel_in_disc,
             n_channels,
             n_conditions,
             device,
             sequence_length_generated=-1,
             hidden_dim=128,
             num_layers=2,
             activation='tanh',
             input_sequence_length=0,
             patch_size=-1,
             **kwargs,
             ):
    # ttsgan-direct pipeline: always build a plain TTSGenerator/TTSDiscriminator pair.
    # No autoencoder-wrapped DecoderGenerator/EncoderDiscriminator path -- that branch
    # was removed here (see openspec/changes/ttsgan-native-multichannel); it remains
    # available on main for anyone using autoencoder-coupled GAN training.
    assert channel_in_disc == n_channels + n_conditions, (
        f"channel_in_disc ({channel_in_disc}) must equal n_channels + n_conditions "
        f"({n_channels} + {n_conditions} = {n_channels + n_conditions})."
    )

    generator = gan_architectures[gan_types['tts'][0]](
        latent_dim=latent_dim_in,
        channels=n_channels,
        seq_len=sequence_length_generated,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        activation=activation,
        num_heads=4,

        # additional TTSGenerator inputs: patch_size
        patch_size=patch_size,
        use_postnet=kwargs.get('use_postnet', False),
    )
    assert generator.channels == n_channels, (
        f"Generator was built with {generator.channels} output channels, expected n_channels={n_channels}."
    )

    discriminator = gan_architectures[gan_types['tts'][1]](
        channels=channel_in_disc,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        seq_len=sequence_length_generated,
        num_heads=4,

        # additional TTSDiscriminator inputs: patch_size
        patch_size=patch_size,
    )

    discriminator2 = None
    secondary_feat_dim = 100 # Default fallback
    if kwargs.get('use_multiscale_dwt_discriminator', False):
        j = kwargs.get('dwt_j', 4)
        discriminator2 = MultiscaleDWTDiscriminator(
            in_channels=channel_in_disc,
            J=j,
            n_classes=1,
            seq_len=kwargs.get('sequence_length'),
            include_high_freq=kwargs.get('multiscale_dwt_high_freq', False)
        ).to(device)
        # Calculation: (hidden_dim // 2) * J * num_streams
        # hidden=64 -> 32 * J * (2 if high_freq else 1)
        hd = 64
        ns = 2 if kwargs.get('multiscale_dwt_high_freq', False) else 1
        secondary_feat_dim = (hd // 2) * j * ns

    # Wrap in StackingDiscriminator only if requested and D2 exists
    if kwargs.get('use_stacking', False) and discriminator2 is not None:
         from eeggan.nn_architecture.models import StackingDiscriminator
         
         # Calculate Primary Dimension
         # TODO (future): derive dynamically from discriminator.[-1].linear.in_features
         # once emb_size becomes a configurable hyperparameter. Currently fixed at 50
         # because TTSDiscriminator is always built with emb_size=50 (see gan_architectures).
         primary_dim = 50
         
         discriminator = StackingDiscriminator(
             primary_disc=discriminator,
             secondary_disc=discriminator2,
             primary_feat_dim=primary_dim,
             secondary_feat_dim=secondary_feat_dim, 
             hidden_dim=64
         ).to(device)
         
         # If stacking, D2 is internal, so we don't return it separately
         return generator, discriminator, None
            
    return generator, discriminator, discriminator2