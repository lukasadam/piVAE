# ---------------------------------------------------------
# TensorFlow piVAE Eager implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Adam
# Email: gm.lukas.adam@gmail.com
# ---------------------------------------------------------

# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from models.vae import vae_encoder, vae_decoder
from models.utils import _phi, _pairwise_squared_distance_matrix, _cross_squared_distance_matrix, _solve_interpolation

class piVAE:
    
    """
    Class containing the rebuild code basis for a prior encoding variational
    autoencoder. piVAE was implemented in TF 2.0.
    """
    
    def __init__(self, config):
        """
        
        Initialization
        
        Parameters:
        config (dict): build config for piVAE

        """
        
        # Store build config 
        self.config = config
        
        # Define array to store all trainable variables here
        self.trainable_variables = []
        
        # Build model then
        self.__build__()
    
    #######################################################################################################################
        
    def __build__(self, init='normal'):
        """
        
        Function that loads standard VAE encoder and decoder modules and initializes weights for the trainable linear basis 

        Parameters:
        init (str): Type of weights initialization

        """
        
        # Load standard vae encoder and decoder from models
        
        self.encoder = vae_encoder(
            input_shape=self.config['encoder_dim']+2, 
            intermediate_dim=self.config['intermediate_dim'], 
            latent_dim=self.config['latent_dim']  
        )
        
        self.decoder = vae_decoder(
            latent_dim=self.config['latent_dim'],
            intermediate_dim=self.config['intermediate_dim'], 
            original_dim=self.config['encoder_dim']+2)
        
        # Now store all trainable variables
        
        self.trainable_variables.extend(self.encoder.trainable_variables)
        self.trainable_variables.extend(self.decoder.trainable_variables)
    
    #######################################################################################################################
    
    def __vae__(self, lb):
            
        """
        
        Standard VAE here

        """
        
        # Encode function proposals
        
        self.z_mean, self.z_log_var, self.z = self.encoder(lb)
        
        # Decode latent sample and reconstruct linear basis
        lb_reconstruct = self.decoder(self.z)
            
        return lb_reconstruct
    
    #######################################################################################################################
       
    def __solve__(self, c, f):
        
        """
        
        Function that solves a linear system from the generated function values and real evaluations. 
        
        Returns: a linear basis abstracted from the equation
        
        """
        
        return _solve_interpolation(c, f, self.config['order'], self.config['regularization_weight'])
        
    #######################################################################################################################
    
    def __mat_op__(self, phi_dists, bias, w, v):
        
        rbf_term = tf.matmul(phi_dists, w)
        
        linear_term = tf.matmul(bias, v)
        
        return rbf_term + linear_term
        
    #######################################################################################################################
    
    def __eval__(self, c, linear_basis, f=None, q=None, training=True):
        
        # These dimensions are set dynamically at runtime.
        b, n, _ = tf.unstack(tf.shape(c), num=3)

        w = linear_basis[:, :n, :]
        v = linear_basis[:, n:, :]
            
        if training:
            # Then calculate pairwise distance between centers
            pairwise_dists = _pairwise_squared_distance_matrix(c)
        
            # Transformed pairwise dists
            phi_dists = _phi(pairwise_dists, order=self.config['order']) 
            
            # Then, compute the contribution from the linear term.
            # Pad query_points with ones, for the bias term in the linear model.
            bias = tf.concat([
                f,
                tf.ones_like(f[..., :1], c.dtype)
            ], 2)
            
        else:
            # First, compute the contribution from the rbf term.
            pairwise_dists = _cross_squared_distance_matrix(q, c)

            phi_dists = _phi(pairwise_dists, order=self.config['order'])

            # Then, compute the contribution from the linear term.
            # Pad query_points with ones, for the bias term in the linear model.
            bias = tf.concat([
                q,
                tf.ones_like(q[..., :1], c.dtype)
            ], 2)

        return self.__mat_op__(
                phi_dists=phi_dists,
                bias=bias,
                w=w,
                v=v
            )
       
      
    #######################################################################################################################
        
    def __train__(self, train_points, train_values, regularization_weight=1.0):
        
        """
        
        Function that is used when training the model. Eager execution is applied. 
        
        Requires:
        train points (np.ndarray): X-axis values

        """
        
        ###########################################
        
        # Squeeze and unsqueeze linear basis
        def squeeze(lb):
            return tf.squeeze(lb, axis=2)
        
        def unsqueeze(lb):
            return tf.cast(tf.expand_dims(lb, axis=2), tf.float64)
        
        ###########################################
        
        if not isinstance(train_points, np.ndarray) and not isinstance(train_values, np.ndarray):
            raise TypeError('training must be performed with numpy array')
        
        # First convert training points to tf
        train_points = tf.convert_to_tensor(train_points)
        train_values = tf.convert_to_tensor(train_values)
        
        ##################################################################
        
        linear_basis = self.__solve__(train_points, train_values)
        
        linear_basis_reconstruct = unsqueeze(self.__vae__(squeeze(linear_basis)))
        
        ##################################################################
        
        # TO-DO: Try to also encode inverse element of vector by decoding a-a = 0 of z-mean
        
        ##################################################################
        
        return linear_basis, linear_basis_reconstruct, self.z_mean, self.z_log_var
    
    
##########################################################################

def pivae_loss(model, x, y=None, regularization_weight=1.0):
    
    def reconstruction_loss(y1, y2, loss_function='mse'):
        
        if loss_function == 'mae':
            loss = tf.reduce_sum(tf.keras.losses.MAE(y1, y2))
        elif loss_function == 'mse':
            loss = tf.reduce_sum(tf.keras.losses.MSE(y1, y2), axis=1)
        elif loss_function == 'msle':
            loss = tf.reduce_sum(tf.keras.losses.MSLE(y1, y2))
        else:
            raise ValueError('no loss function given!')
        return loss
    
    def kl_divergence_loss(z_mean, z_log_var):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = tf.cast(kl_loss, tf.float64)
        return kl_loss
    
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    
    lb, lb_reconstruct, z_mean, z_log_var = model.__train__(x, y) 
    
    # Reconstruction loss - linear basis
    rl1 = reconstruction_loss(lb, lb_reconstruct)
   
    rl2 = reconstruction_loss(y, model.__eval__(c=x, linear_basis=lb_reconstruct, f=x))
    
    # KL divergence loss
    kl_loss = kl_divergence_loss(z_mean, z_log_var)

    # Total loss = 50% rec + 50% KL divergence loss
       
    loss = K.mean(rl1 + rl2 + kl_loss) 
    
    return loss

############################################################################

def pivae_grad(model, x, y, trainable_variables):
    with tf.GradientTape() as tape:
        loss_value = pivae_loss(model, x, y)
    return loss_value, tape.gradient(loss_value, trainable_variables)