import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from eeggan.nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder
from eeggan.nn_architecture.models import TTSGenerator, TTSDiscriminator, DecoderGenerator, EncoderDiscriminator, MultiscaleDWTDiscriminator


gan_architectures = {
        'TTSGenerator': lambda seq_len, hidden_dim, patch_size, channels, latent_dim, num_layers, num_heads, **kwargs: TTSGenerator(seq_len, patch_size, channels, 1, latent_dim, 10, num_layers, num_heads, 0.5, 0.5),
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
             autoencoder='',
             **kwargs,
             ):
    if autoencoder == '':
        # no autoencoder defined -> use transformer GAN
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
    else:
        # initialize an autoencoder-GAN

        # initialize the autoencoder
        ae_dict = torch.load(autoencoder, map_location=torch.device('cpu'), weights_only=False)
        if ae_dict['configuration']['target'] == 'channels':
            ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_CHANNELS
            autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
        elif ae_dict['configuration']['target'] == 'time':
            ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_TIMESERIES
            autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
        elif ae_dict['configuration']['target'] == 'full':
            autoencoder = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=2).to(device)
            autoencoder.model_1 = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=1).to(device)
            autoencoder.model_1.eval()
        else:
            raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
        consume_prefix_in_state_dict_if_present(ae_dict['model'], 'module.')
        autoencoder.load_state_dict(ae_dict['model'])
        # freeze the autoencoder
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.eval()
        
        # adjust generator output_dim to match the output_dim of the autoencoder
        if autoencoder.target == 2: # TARGET_BOTH / FULL
             n_channels = autoencoder.linear_dec_in_channels.weight.shape[1]
             sequence_length_generated = autoencoder.model_1.linear_dec_in_timeseries.weight.shape[1]
        elif autoencoder.target == 1: # TARGET_TIMESERIES
             # n_channels = autoencoder.output_dim_2 # Do not overwrite n_channels for Time-series AE
             sequence_length_generated = autoencoder.linear_dec_in.weight.shape[1]
        else: # TARGET_CHANNELS
             n_channels = autoencoder.linear_dec_in.weight.shape[1]
             sequence_length_generated = autoencoder.output_dim_2

        # adjust discriminator input_dim to match the output_dim of the autoencoder
        channel_in_disc = n_channels + n_conditions

        generator = DecoderGenerator(
            generator=gan_architectures[gan_types['tts'][0]](
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
            ),
            decoder=autoencoder
        )

        discriminator = EncoderDiscriminator(
            discriminator=gan_architectures[gan_types['tts'][1]](
                channels=channel_in_disc,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
                seq_len=sequence_length_generated,
                num_heads=4,

                # additional TTSDiscriminator inputs: patch_size
                patch_size=patch_size,
            ),
            encoder=autoencoder
        )

        if isinstance(discriminator, EncoderDiscriminator) and isinstance(generator, DecoderGenerator) and input_sequence_length == 0:
            # if input_sequence_length is 0, do not encode the discriminator input during training
            # discriminator.encode_input(False)
            pass

    discriminator2 = None
    secondary_feat_dim = 100 # Default fallback
    if kwargs.get('use_multiscale_dwt_discriminator', False):
        discriminator2 = MultiscaleDWTDiscriminator(
            in_channels=channel_in_disc,
            J=4,  # 4 levels of decomposition
            n_classes=1,
            seq_len=kwargs.get('sequence_length'),
            include_high_freq=kwargs.get('multiscale_dwt_high_freq', False)
        ).to(device)
        # Calculation: (hidden_dim // 2) * J * num_streams
        # hidden=64 -> 32 * 4 * (2 if high_freq else 1) = 128 or 256
        hd = 64
        j = 4
        ns = 2 if kwargs.get('multiscale_dwt_high_freq', False) else 1
        secondary_feat_dim = (hd // 2) * j * ns

    # Wrap in StackingDiscriminator only if requested and D2 exists
    if kwargs.get('use_stacking', False) and discriminator2 is not None:
         from eeggan.nn_architecture.models import StackingDiscriminator
         
         # Calculate Primary Dimension
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