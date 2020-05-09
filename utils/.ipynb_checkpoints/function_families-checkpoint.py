# ---------------------------------------------------------
# Data Loader & UTILS
# Licensed under The MIT License [see LICENSE for details]
# Written by Lukas Adam
# Email: gm.lukas.adam@gmail.com
# ---------------------------------------------------------

# Imports
import numpy as np
import pandas as pd
import random

# Define noise
def noise(x, loc=0, std=1):
    """Add gaussian noise"""
    noise = np.random.normal(loc, std, x.shape[0])
    noise = np.expand_dims(noise, axis=1)
    return np.sum([x, noise], axis=0)

########################

# Put functions here

def linear_functions(c):
    linear = lambda x: noise(c*x, std=0.01)
    linear.__name__ = str(round(3, 3)) + '_lin'
    return linear

def square_functions(c):
    square = lambda x: noise(c*np.power(x, 2))
    square.__name__ = str(round(c, 3)) + '_square'
    return square

########################

def random_constant(x_low=-5, x_high=5):
    return random.uniform(x_low, x_high)

########################

def generate_function_family(function, sample_size=20):
    '''Input: Must be lambda function'''
    
    return [function(random_constant()) for i in range(sample_size)]
