import tensorflow as tf
import tensorflow_compression as tfc
from models.cvc_block import VitBlock
from models.channel import AWGNChannel

__all__ = ['SemViT', 'SemViT_clip', 'SemViT_power', 'SemViT_OFDM_PAPR']

class SemViT(tf.keras.Model):
    def __init__(self, filters=256, num_symbols=512, snrdB=25, channel='AWGN', **kwargs):
        super().__init__()

        self.enc0 = tfc.SignalConv2D(
            filters=filters,
            kernel_support=9,
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=tf.keras.layers.PReLU(shared_axes=[1, 2])
        )
        self.enc1 = tfc.SignalConv2D(
            filters=filters,
            kernel_support=5,
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=tf.keras.layers.PReLU(shared_axes=[1, 2])
        )
        self.enc21 = VitBlock(num_heads=8, head_size=32, spatial_size=8)
        self.enc22 = VitBlock(num_heads=8, head_size=32, spatial_size=8)
        self.enc23 = VitBlock(num_heads=8, head_size=32, spatial_size=8)
        self.enc_proj = tf.keras.layers.Conv2D(
            filters=num_symbols // 8 // 8 * 2,
            kernel_size=1
        )

        if channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        else:
            self.channel = tf.keras.layers.Lambda(lambda x: x)

        self.dec01 = VitBlock(num_heads=8, head_size=32, spatial_size=8)
        self.dec02 = VitBlock(num_heads=8, head_size=32, spatial_size=8)
        self.dec03 = VitBlock(num_heads=8, head_size=32, spatial_size=8)
        self.dec1 = tfc.SignalConv2D(
            filters=filters,
            kernel_support=5,
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=tf.keras.layers.PReLU(shared_axes=[1, 2])
        )
        self.dec2 = tfc.SignalConv2D(
            filters=filters,
            kernel_support=9,
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=tf.keras.layers.PReLU(shared_axes=[1, 2])
        )
        self.dec_proj = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            activation='sigmoid'
        )
        self.num_symbols = num_symbols
        self.mse_metric = tf.keras.losses.MeanSquaredError()

    
    def call(self, x):
        # Encode
        input_image = x        
        
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc21(x)
        x = self.enc22(x)
        x = self.enc23(x)
        symbols = self.enc_proj(x)

        # I/Q transmission
        b, h, w, c = symbols.shape
        symbols_corrupted = self.channel(tf.reshape(symbols, (-1, h*w*c//2, 2)))
        b, c, _ = symbols_corrupted.shape
        symbols_corrupted = tf.reshape(symbols_corrupted, (-1, 8, 8, c*2//64))

        # Decode
        x = self.dec01(symbols_corrupted)
        x = self.dec02(x)
        x = self.dec03(x)
        x = self.dec1(x)
        x = self.dec2(x)
        pred = self.dec_proj(x)

        # Neural Learning
        mse_loss = self.mse_metric(input_image, pred)
        loss = mse_loss
        psnr = tf.image.psnr(input_image, pred, max_val=1)
        metric = {'psnr':psnr}
        return pred, loss, metric
    
    
    def train_step(self, x):
        x, _ = x
        with tf.GradientTape() as tape:
            _, loss, metric = self(x, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss.update_state(loss)
        self.psnr.update_state(metric['psnr'])
        return {m.name: m.result() for m in [self.loss, self.psnr]}

    def test_step(self, x):
        x, _ = x
        _, loss, metric = self(x, training=True)
        self.loss.update_state(loss)
        self.psnr.update_state(metric['psnr'])
        return {m.name: m.result() for m in [self.loss, self.psnr]}  
    
    def predict_step(self, x):
        x, _ = x
        pred, _, _ = self(x, training=False)
        return pred

    def compile(self, **kwargs):
        del kwargs['loss']
        del kwargs['metrics']
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.psnr = tf.keras.metrics.Mean(name="psnr")
       
        
class SemViT_clip(SemViT):
    def __init__(self, mean_coeff, std_coeff, clip_limit, **kwargs):
        super().__init__(**kwargs)
        self.mean_coeff = mean_coeff
        self.std_coeff = std_coeff
        self.clip_limit = clip_limit
    
    
    def call(self, x, training=False):
        # Encode
        input_image = x        
        
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc21(x)
        x = self.enc22(x)
        x = self.enc23(x)
        x = self.enc_proj(x)
        
        # Clipping
        symbols = tf.clip_by_value(x, -self.clip_limit, self.clip_limit)
        
        # I/Q transmission
        b, h, w, c = symbols.shape
        symbols = tf.reshape(symbols, (-1, h*w*c//2, 2))
        signal_power = symbols[:,:,0]**2 + symbols[:,:,1]**2
        symbols_corrupted = self.channel(symbols)
        b, c, _ = symbols_corrupted.shape
        symbols_corrupted = tf.reshape(symbols_corrupted, (-1, 8, 8, c*2//64))

        # Decode
        x = self.dec01(symbols_corrupted)
        x = self.dec02(x)
        x = self.dec03(x)
        x = self.dec1(x)
        x = self.dec2(x)
        pred = self.dec_proj(x)

        # Neural Learning
        goal_pow = tf.ones([tf.shape(x)[0]]) * 2
        mse_loss = self.mse_metric(input_image, pred)
        power_mean = tf.reduce_mean(signal_power, axis=1)
        mean_loss = tf.abs(power_mean - goal_pow)
        power_std = tf.math.reduce_std(signal_power, axis=1)
        std_loss = power_std
        loss = mse_loss + self.mean_coeff*mean_loss + self.std_coeff*std_loss
        
        psnr = tf.image.psnr(input_image, pred, max_val=1)
        papr = tf.reduce_max(signal_power, axis=1) / power_mean
        metric = {'psnr':psnr, 'power_mean':power_mean, 'power_std':power_std, 'papr':papr}
        
        return pred, loss, metric
    
    
    def train_step(self, x):
        x, _ = x
        with tf.GradientTape() as tape:
            _, loss, metric = self(x, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss.update_state(loss)
        self.psnr.update_state(metric['psnr'])
        self.power_mean.update_state(metric['power_mean'])
        self.power_std.update_state(metric['power_std'])
        self.papr.update_state(metric['papr'])
        return {m.name: m.result() for m in [self.loss, self.psnr, self.power_mean, self.power_std, self.papr]}

    def test_step(self, x):
        x, _ = x
        _, loss, metric = self(x, training=True)
        self.loss.update_state(loss)
        self.psnr.update_state(metric['psnr'])
        self.power_mean.update_state(metric['power_mean'])
        self.power_std.update_state(metric['power_std'])
        self.papr.update_state(metric['papr'])
        return {m.name: m.result() for m in [self.loss, self.psnr, self.power_mean, self.power_std, self.papr]}
    
    def predict_step(self, x):
        x, _ = x
        pred, _, _ = self(x, training=False)
        return pred

    def compile(self, **kwargs):
        super().compile(
            **kwargs,
        )
        self.power_mean = tf.keras.metrics.Mean(name="power_mean")
        self.power_std = tf.keras.metrics.Mean(name="power_std")
        self.papr = tf.keras.metrics.Mean(name="papr")
        

class SemViT_power(SemViT):
    def __init__(self, mean_coeff, std_coeff, papr_coeff=0, clip_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_coeff = mean_coeff
        self.std_coeff = std_coeff
        self.papr_coeff = papr_coeff
        self.clip_limit = clip_limit
        self.norm = tf.Variable(tf.ones((1,)), trainable=True, name='normalizer')
    
    def call(self, x, training=False):
        # Encode
        input_image = x        
        
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc21(x)
        x = self.enc22(x)
        x = self.enc23(x)
        symbols = self.enc_proj(x) / self.norm
                  
        # Clipping
        if self.clip_limit is not None:
            # signal_norm = tf.reduce_mean(symbols ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5
            # clip = signal_norm * self.clip_limit
            # symbols = tf.clip_by_value(symbols, -clip, clip)
            symbols = tf.clip_by_value(symbols, -self.clip_limit, self.clip_limit)
        
        # I/Q transmission
        b, h, w, c = symbols.shape
        symbols = tf.reshape(symbols, (-1, h*w*c//2, 2))
        signal_power = symbols[:,:,0]**2 + symbols[:,:,1]**2
        symbols_corrupted = self.channel(symbols)
        b, c, _ = symbols_corrupted.shape
        symbols_corrupted = tf.reshape(symbols_corrupted, (-1, 8, 8, c*2//64)) * self.norm

        # Decode
        x = self.dec01(symbols_corrupted)
        x = self.dec02(x)
        x = self.dec03(x)
        x = self.dec1(x)
        x = self.dec2(x)
        pred = self.dec_proj(x)

        # Neural Learning
        mse_loss = self.mse_metric(input_image, pred)
        mean_loss = tf.abs(tf.reduce_mean(signal_power) - tf.ones((1,)))
        power_std = tf.math.reduce_std(signal_power, axis=1)
        std_loss = power_std
        power_mean = tf.reduce_mean(signal_power, axis=1)
        papr_loss = tf.reduce_max(signal_power, axis=1) / power_mean
        loss = mse_loss + self.mean_coeff*mean_loss + self.std_coeff*std_loss + self.papr_coeff*papr_loss
        
        psnr = tf.image.psnr(input_image, pred, max_val=1)
        metric = {'psnr':psnr, 'power_mean':power_mean, 'power_std':power_std, 'papr':papr_loss, 'norm_val':self.norm}
        
        return pred, loss, metric
    
    
    def state_manager(self, loss, metric):
        self.loss.update_state(loss)
        self.psnr.update_state(metric['psnr'])
        self.power_mean.update_state(metric['power_mean'])
        self.power_std.update_state(metric['power_std'])
        self.papr.update_state(metric['papr'])
        self.norm_val.update_state(metric['norm_val'])
        return {m.name: m.result() for m in [self.loss, self.psnr, self.power_mean, self.power_std, self.papr, self.norm_val]}
    
    def train_step(self, x):
        x, _ = x
        with tf.GradientTape() as tape:
            _, loss, metric = self(x, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self.state_manager(loss, metric)

    def test_step(self, x):
        x, _ = x
        _, loss, metric = self(x, training=True)
        return self.state_manager(loss, metric)
    
    def predict_step(self, x):
        x, _ = x
        pred, _, _ = self(x, training=False)
        return pred

    def compile(self, **kwargs):
        super().compile(
            **kwargs,
        )
        self.power_mean = tf.keras.metrics.Mean(name="power_mean")
        self.power_std = tf.keras.metrics.Mean(name="power_std")
        self.papr = tf.keras.metrics.Mean(name="papr")
        self.norm_val = tf.keras.metrics.Mean(name="norm_val")
        

class SemViT_power_enc(SemViT):
    def __init__(self, clip_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.Variable(tf.ones((1,)), trainable=True, name='normalizer')
        self.clip_limit = clip_limit 
    
    def call(self, x, training=False):
        # Encode     
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc21(x)
        x = self.enc22(x)
        x = self.enc23(x)
        symbols = self.enc_proj(x) / self.norm
                  
        # Clipping
        if self.clip_limit is not None:
            symbols = tf.clip_by_value(symbols, -self.clip_limit, self.clip_limit)
        
        # Complex Encoding
        b, h, w, c = symbols.shape
        symbols = tf.reshape(symbols, (-1, h*w*c//2, 2))
        i = symbols[:,:,0]
        q = symbols[:,:,1]
        symbols = tf.complex(i, q)
    
        return symbols
    
  
class SemViT_power_dec(SemViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.Variable(tf.ones((1,)), trainable=True, name='normalizer')
    
    def call(self, x, training=False):
        symbols = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        b, c, _ = symbols.shape
        symbols = tf.reshape(symbols, (-1, 8, 8, c*2//64)) * self.norm

        # Decode
        x = self.dec01(symbols)
        x = self.dec02(x)
        x = self.dec03(x)
        x = self.dec1(x)
        x = self.dec2(x)
        pred = self.dec_proj(x)

        return pred
    

class SemViT_OFDM_PAPR(SemViT):
    def __init__(self, papr_coeff=0, **kwargs):
        super().__init__(**kwargs)
        self.papr_coeff = papr_coeff
        self.FFT_size = 64
    
    def call(self, x, training=False):
        # Encode
        input_image = x        
        
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc21(x)
        x = self.enc22(x)
        x = self.enc23(x)
        symbols = self.enc_proj(x)
                  
        # I/Q transmission
        b, h, w, c = symbols.shape
        symbols = tf.reshape(symbols, (-1, h*w*c//2, 2))
        IQ = tf.complex(symbols[:,:,0], symbols[:,:,1])
        symbols_corrupted = self.channel(symbols)
        b, c, _ = symbols_corrupted.shape
        symbols_corrupted = tf.reshape(symbols_corrupted, (-1, 8, 8, c*2//64))

        # Decode
        x = self.dec01(symbols_corrupted)
        x = self.dec02(x)
        x = self.dec03(x)
        x = self.dec1(x)
        x = self.dec2(x)
        pred = self.dec_proj(x)

        # Neural Learning
        if training:
            mse = tf.reduce_mean((input_image - pred)**2, axis=[1, 2, 3])
            IQ = tf.reshape(IQ, (-1, self.num_symbols//self.FFT_size, self.FFT_size))
            IQ_t = tf.signal.ifft(IQ)
            IQ_pow = tf.abs(IQ_t)**2
            papr = tf.reduce_max(IQ_pow, axis=2) / tf.reduce_mean(IQ_pow, axis=2)
            loss = tf.reduce_mean(mse + self.papr_coeff*tf.reduce_mean(papr, axis=1))
            
            psnr = tf.image.psnr(input_image, pred, max_val=1)
            metric = {'psnr':psnr, 'mse':mse, 'papr':papr}
            return pred, loss, metric
        else:
            return pred
    
    
    def state_manager(self, loss, metric):
        self.loss.update_state(loss)
        self.psnr.update_state(metric['psnr'])
        self.mse.update_state(metric['mse'])
        self.papr.update_state(metric['papr'])
        return {m.name: m.result() for m in [self.loss, self.psnr, self.mse, self.papr]}
    
    def train_step(self, x):
        x, _ = x
        with tf.GradientTape() as tape:
            _, loss, metric = self(x, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self.state_manager(loss, metric)

    def test_step(self, x):
        x, _ = x
        _, loss, metric = self(x, training=True)
        return self.state_manager(loss, metric)

    def compile(self, **kwargs):
        super().compile(
            **kwargs,
        )
        self.mse = tf.keras.metrics.Mean(name="mse")
        self.papr = tf.keras.metrics.Mean(name="papr")
    