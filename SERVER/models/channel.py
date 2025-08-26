import tensorflow as tf


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
    
    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"

        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        y = x + n
        return y

    def get_config(self):
        config = super().get_config()
        return config
    
    
class REALChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, amp_fluc=None, phase_fluc=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.amp_fluc = amp_fluc
        self.phase_fluc = phase_fluc
    
    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"

        i = x[:,:,0]
        q = x[:,:,1]
                        
        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        y = x + n
                
        iq = tf.complex(y[:,:,0], y[:,:,1])
        # amp_noise = tf.random.normal(shape=(1,), mean=1, stddev=self.amp_fluc)
        # phase_noise = tf.random.normal(shape=(1,), mean=0, stddev=self.phase_fluc)
        amp_noise = tf.random.normal(shape=tf.shape(iq), mean=1, stddev=self.amp_fluc)
        phase_noise = tf.random.normal(shape=tf.shape(iq), mean=0, stddev=self.phase_fluc)
        iq *= tf.complex(amp_noise, 0.,)
        iq *= tf.exp(tf.complex(0., phase_noise))
        iq = tf.expand_dims(iq, axis=2)
        y = tf.concat([tf.math.real(iq), tf.math.imag(iq)], axis=2)
                
        return y

    def get_config(self):
        config = super().get_config()
        return config
    
    
class RICIANChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, shape_param=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.sigma = (1 / (2 * shape_param))**0.5
    
    def call(self, x, return_channel=False):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"

        i = x[:,:,0]
        q = x[:,:,1]
        x = tf.complex(i, q)
                        
        h_i = tf.random.normal(tf.shape(i), mean=2**0.5, stddev=self.sigma)
        h_q = tf.random.normal(tf.shape(q), mean=2**0.5, stddev=self.sigma)
        h = tf.complex(h_i, h_q)
        hx = h * x
        
        sig_power = tf.abs(hx)**2
        no = tf.math.sqrt(sig_power/(2*self.snr))
        n = tf.complex(
            tf.random.normal(tf.shape(i), mean=0, stddev=no),
            tf.random.normal(tf.shape(q), mean=0, stddev=no)
        )
        y = hx + n
        
        y = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        if return_channel:
            return y, tf.stack([h_i, h_q], axis=-1)
        else:
            return y

    def get_config(self):
        config = super().get_config()
        return config
