# ---------------------------------------------------------
# TensorFlow piVAE Encoder & Decoder Models
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Adam
# Email: gm.lukas.adam@gmail.com
# ---------------------------------------------------------

# Imports
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K

tfd = tfp.distributions

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def _sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * (z_log_var)) * epsilon

##################################################

def vae_encoder(input_shape, intermediate_dim, latent_dim, sampling=_sampling):
    inputs = Input(shape=input_shape, name='encoder_input')
    
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    return encoder

##################################################

def vae_decoder(latent_dim, intermediate_dim, original_dim):
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim)(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    return decoder
