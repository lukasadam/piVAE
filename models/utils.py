# ---------------------------------------------------------
# TensorFlow piVAE Utils
# Licensed under The MIT License [see LICENSE for details]
# Written Lukas Adam
# Email: gm.lukas.adam@gmail.com
# ---------------------------------------------------------

import tensorflow as tf

def _pairwise_squared_distance_matrix(x):
    """Pairwise squared distance among a (batch) matrix's rows (2nd dim).
    This saves a bit of computation vs. using
    _cross_squared_distance_matrix(x,x)
    Args:
      x: `[batch_size, n, d]` float `Tensor`
    Returns:
      squared_dists: `[batch_size, n, n]` float `Tensor`, where
      squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2
    """

    x_x_transpose = tf.matmul(x, x, adjoint_b=True)
    x_norm_squared = tf.linalg.diag_part(x_x_transpose)
    x_norm_squared_tile = tf.expand_dims(x_norm_squared, 2)

    # squared_dists[b,i,j] = ||x_bi - x_bj||^2 =
    # = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared_tile - 2 * x_x_transpose + tf.transpose(
        x_norm_squared_tile, [0, 2, 1])

    return squared_dists


def _cross_squared_distance_matrix(x, y):
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).
    Computes the pairwise distances between rows of x and rows of y
    Args:
      x: [batch_size, n, d] float `Tensor`
      y: [batch_size, m, d] float `Tensor`
    Returns:
      squared_dists: [batch_size, n, m] float `Tensor`, where
      squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
    """
    x_norm_squared = tf.reduce_sum(tf.square(x), 2)
    y_norm_squared = tf.reduce_sum(tf.square(y), 2)

    # Expand so that we can broadcast.
    x_norm_squared_tile = tf.expand_dims(x_norm_squared, 2)
    y_norm_squared_tile = tf.expand_dims(y_norm_squared, 1)

    x_y_transpose = tf.matmul(x, y, adjoint_b=True)

    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 =
    # x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = (
        x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile)

    return squared_dists

def _phi(r, order=2, epsilon=0.0000000001):
        """Coordinate-wise nonlinearity used to define the order of the
        interpolation.
        See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.
        Args:
          r: input op
          order: interpolation order
        Returns:
          phi_k evaluated coordinate-wise on r, for k = r
        """

        # using epsilon prevents log(0), sqrt0), etc.
        # sqrt(0) is well-defined, but its gradient is not
        with tf.name_scope('phi'):
            if order == 1:
                r = tf.maximum(r, epsilon)
                r = tf.sqrt(r)
                return r
            elif order == 2:
                return 0.5 * r * tf.math.log(tf.maximum(r, epsilon))
            elif order == 4:
                return 0.5 * tf.square(r) * tf.math.log(tf.maximum(r, epsilon))
            elif order % 2 == 0:
                r = tf.maximum(r, epsilon)
                return 0.5 * tf.pow(r, 0.5 * order) * tf.math.log(r)
            else:
                r = tf.maximum(r, epsilon)
                return tf.pow(r, 0.5 * order)

def _solve_interpolation(c, f, order=1, regularization_weight=1.0):
    # These dimensions are set dynamically at runtime.
    b, n, _ = tf.unstack(tf.shape(c), num=3)
        
    d = c.shape[-1]
    if d is None:
        raise ValueError('The dimensionality of the input points (d) must be '
                         'statically-inferrable.')
    
    k = f.shape[-1]
    if k is None:
        raise ValueError('The dimensionality of the output values (k) must be '
                         'statically-inferrable.')
        
    # Then calculate pairwise distance between centers
    pairwise_dists = _pairwise_squared_distance_matrix(c)
        
    # Transformed pairwise dists
    matrix_a = _phi(pairwise_dists, order=order) 
        
    ###########
        
    if regularization_weight > 0:
        batch_identity_matrix = tf.expand_dims(tf.eye(n, dtype=c.dtype), 0)
        matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term
    # in the linear model.
    ones = tf.ones_like(c[..., :1], dtype=c.dtype)
    matrix_b = tf.concat([c, ones], 2)  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = tf.concat(
            [matrix_a, tf.transpose(matrix_b, [0, 2, 1])], 1)

    num_b_cols = matrix_b.get_shape()[2]  # d + 1
    lhs_zeros = tf.zeros([b, num_b_cols, num_b_cols], c.dtype)
    right_block = tf.concat([matrix_b, lhs_zeros],
                                1)  # [b, n + d + 1, d + 1]
    lhs = tf.concat([left_block, right_block],
                        2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = tf.zeros([b, d + 1, k], c.dtype)
    rhs = tf.concat([f, rhs_zeros], 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    with tf.name_scope('solve_linear_system'):
        linear_basis = tf.cast(tf.linalg.solve(lhs, rhs), tf.float64)
           
    return linear_basis