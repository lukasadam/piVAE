# ---------------------------------------------------------
# Data Loader & UTILS
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Adam
# Email: gm.lukas.adam@gmail.com
# ---------------------------------------------------------

# Imports
import numpy as np
import pandas as pd

# Define noise
def noise(x, loc=0, std=1):
    """Add gaussian noise"""
    noise = np.random.normal(loc, std, x.shape[0])
    noise = np.expand_dims(noise, axis=1)
    return np.sum([x, noise], axis=0)

###################################################

# Define Onehot Encoding
def onehot(x):
    x = pd.Series(x)
    x = pd.get_dummies(x)
    return x.values

###################################################

# Define sample operations
def sample_index(x):
    return np.random.randint(x, size=1)[0]

def sample_uniform(x_low, x_high, e_dim):
    return np.random.uniform(x_low, x_high, e_dim)[:, np.newaxis]
    
def sample_function(x, functions):
    index = sample_index(len(functions))
    function_eval = functions[index](x)
    return (x, index, function_eval)
        
###################################################
    
def Polyharmonic_Spline_Dataset(functions, config):
        
    # Store sampled functions and indices here
    X, Y, S = [[] for i in range(3)]
            
    # Define training X & Y and test X
    x = sample_uniform(config['x_low'], config['x_high'], config['encoder_dim'])
      
    for batch in range(config['batch_size']):
        x, index, function_eval = sample_function(x, functions) 
        X.append(x)
        Y.append(function_eval)
        S.append(index)
    
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    
    return X, Y, S
            
