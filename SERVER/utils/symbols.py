import tensorflow as tf

__all__ = ['to_complex_symbols', 'to_real_values']

def to_complex_symbols(x):
    i = x[:,:,0]
    q = x[:,:,1]
    return tf.complex(i, q)

def to_real_values(x):
    i = tf.math.real(x)
    q = tf.math.imag(x)
    return tf.stack([i, q], axis=-1)