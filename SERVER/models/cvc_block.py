import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np

def gather_by_index(x, indicies):
  '''
  x: image with shape (B, H, W, C)
  indicies: tensor with shape (B, N, 2)
  return: tensor with shape (B, N, K, K)
  k: kernel size (odd number)
  '''
  # Only differentiable w.r.t. x (not indicies)
  eps = np.finfo(dtype=np.float32).eps
  b, h, w, c = x.shape
  b, n, _ = indicies.shape
  indicies = tf.cast(tf.round(indicies + eps), dtype=tf.int32)
  # (B, N, 2)

  patches = tf.gather_nd(x, indicies, batch_dims=1)
  patches = tf.reshape(patches, (-1, n, c))

  return patches


class MLP(tf.keras.layers.Layer):
    def __init__(self, out_features, expansion_coeff=4):
        super().__init__()

        self.fc1 = tf.keras.layers.Dense(
            out_features * expansion_coeff
        )
        self.gelu = tf.nn.gelu
        self.fc2 = tf.keras.layers.Dense(
            out_features
        )
    
    def call(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class RelativeMHSA(tf.keras.layers.Layer):
    '''
    Implements multihead attention 
    with Swin-like learnable 2d relative positional encoding
    '''
    def __init__(self, num_heads, dim_head, spatial_size, use_pos_emb=True):
        '''
        num_heads: the number of heads
        dim_head: channel dimensions per head
        spatial_size: height/width of the input
        query/key/value shape: (b, h, w, c) where h == w 
        '''
        super().__init__()

        assert num_heads != 0, "num_heads should be nonzero"

        self.dim_head = dim_head
        self.num_heads = num_heads

        self.qkv = tf.keras.layers.Conv2D(
            filters=dim_head * 3,
            kernel_size=1
        )

        self.head_transform = tf.keras.layers.Conv2D(
            filters=dim_head*num_heads,
            kernel_size=1
        )

        self.use_pos_emb = use_pos_emb

        if use_pos_emb:
            # build rel. pos parameter and bias index here
            h = spatial_size
            pos_emb_idx_horizontal = tf.tile(tf.constant(
                [range(i, i+h) for i in range(0, -h, -1)]),
                multiples=[h, h]
            )

            pos_emb_idx_vertical = tf.repeat(
                tf.repeat(
                    tf.constant([range(i, i+h)
                                for i in range(0, -h, -1)]),
                    repeats=h,
                    axis=0
                ),
                repeats=h,
                axis=-1
            )

            pos_emb_idx = (2*h-1) * (pos_emb_idx_vertical + h - 1) + \
                        (pos_emb_idx_horizontal + h - 1)

            self.pos_emb_idx = pos_emb_idx

            initializer = tf.keras.initializers.HeNormal()
            self.learned_pos_emb = tf.Variable(
                initializer(shape=((2*h-1)**2,))
            )

            # initializer = tf.keras.initializers.HeNormal()
            # self.learned_pos_emb_2 = tf.Variable(
            #     initializer(shape=((2*h-1)**2,))
            # )


    def call(self, x, image_coordinates=None):
        b, h, w, c = x.shape
        m = self.num_heads

        assert c % m == 0, "channel dimension should be divisible " \
               f"with number of heads, but c={c} and m={m} found"
        d_h = c//m

        # [b, h, w, c] to [b, m, h, w, c//m]
        x = tf.reshape(x, (-1, h, w, m, d_h))
        x = tf.transpose(x, (0, 3, 1, 2, 4))

        x = self.qkv(x)
        x = tf.reshape(x, (-1, h*w, self.dim_head, 3))
        q = x[:, :, :, 0]
        k = x[:, :, :, 1]
        v = x[:, :, :, 2]

        # normalize with sqrt(d)
        q = q / tf.sqrt(tf.constant(self.dim_head, tf.float32))

        # attention map computation; q, k: (b*m, h*w, d_h)
        att_map = tf.einsum('bic,bjc->bij', q, k)

        if self.use_pos_emb:
            # image_coordinates: (b, h, w, 2)
            if image_coordinates is not None:
                xs = image_coordinates[:, :, :, 0] # (0...31, 0...31, ..., 0...31)
                ys = image_coordinates[:, :, :, 1] # (0...0, 1...1, ..., 31...31)
                # (b, h, w)

                # tf.print(xs.shape, pos_emb_idx[0, 0, :])

                xs_diff = tf.reshape(xs, (-1, 1, h*w)) - tf.reshape(xs, (-1, h*w, 1))
                ys_diff = tf.reshape(ys, (-1, 1, h*w)) - tf.reshape(ys, (-1, h*w, 1))

                pos_emb_idx = tf.reshape(
                    tf.stack((xs_diff+h-1, ys_diff+h-1), axis=-1),
                    (-1, h*w*h*w, 2)
                )

                # when image_coordinates are not given, default pos encoding indicies are equivalent to:
                # pos_emb_idx = (ys_diff+h-1) + (xs_diff+h-1) * (2*h-1)
                # when image_coordinates are unit grid indicies

                grid = tf.reshape(self.learned_pos_emb, (1, 2*h-1, 2*w-1, 1))

                # broadcasting trick
                grid = grid + 0 * tf.reshape(pos_emb_idx[:, 0, 0], (-1, 1, 1, 1))

                # pos_emb = tf.reshape(
                #     tfa.image.interpolate_bilinear(
                #         grid,
                #         pos_emb_idx,
                #     ),
                #     (-1, h*w, h*w)
                # )
                pos_emb = tf.reshape(
                    gather_by_index(
                        grid,
                        pos_emb_idx
                    ),
                    (-1, h*w, h*w)
                )
                # pos_emb should be reshaped regarding multi-head
                # (b, hw, hw) => (b, m, hw, hw) => (bm, hw, hw)

                att_map = tf.reshape(att_map, (-1, m, h*w, h*w))
                pos_emb = tf.expand_dims(pos_emb, axis=1)
                att_map = att_map + pos_emb
                att_map = tf.reshape(att_map, (-1, h*w, h*w))

                # pos_emb_unit_grid = tf.gather(self.learned_pos_emb, self.pos_emb_idx)
                
                # att_map = att_map + pos_emb_unit_grid
            else:
                # add rel. pos. encoding to attention map
                pos_emb = tf.gather(self.learned_pos_emb, self.pos_emb_idx)
                att_map = att_map + pos_emb

        att_map = tf.nn.softmax(att_map)
        
        v = tf.reshape(v, (-1, h*w, self.dim_head))
        v = tf.einsum('bij,bjc->bic', att_map, v)

        # [b, m, h, w, c//m] to [b, h, w, c]
        v = tf.reshape(v, (-1, m, h, w, c//m))
        v = tf.transpose(v, (0, 2, 3, 1, 4))
        v = tf.reshape(v, (-1, h, w, c))

        v = self.head_transform(v)
        return v


class VitBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size,
                 spatial_size, stride=1, use_pos_emb=True):
        '''
        num_heads: the number of heads
        head_size: channel dimensions per head
        spatial_size: height/width of the input
                      (before downsampling)
        patchmerge: (boolean) 1/2 downsampling before MHSA
        '''
        super().__init__()

        d_out = num_heads * head_size
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.patchmerge = tf.keras.layers.Conv2D(
            filters=d_out,
            kernel_size=stride,
            strides=stride,
        )
        spatial_size //= stride

        self.mhsa = RelativeMHSA(
            num_heads=num_heads,
            dim_head=head_size,
            spatial_size=spatial_size,
            use_pos_emb=use_pos_emb
        )

        self.ln2 = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(d_out)

    def call(self, x, image_coordinates=None):
        x = self.patchmerge(x)
        x = self.ln1(x)
        x_residual = x

        x = self.mhsa(x, image_coordinates) 
        x = tf.add(x, x_residual)
        
        x_residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = tf.add(x, x_residual)

        return x
    

class DecodeLossWrapper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, image_warp, pred_warp, name='MSEloss', rate=10, loss=False):
        if loss:
            self.add_loss(rate * self.loss(image_warp, pred_warp))
        self.add_metric(self.loss(image_warp, pred_warp), name)
        return None


class SemViT_Encoder_Only(tf.keras.Model):
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
    
    def call(self, x):
        # Encode
        input_image = x        
        
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc21(x)
        x = self.enc22(x)
        x = self.enc23(x)
        symbols = self.enc_proj(x)

        b, h, w, c = symbols.shape
        return tf.reshape(symbols, (-1, h*w*c//2, 2))

    
class SemViT_Decoder_Only(tf.keras.Model):
    def __init__(self, filters=256, num_symbols=512, snrdB=25, channel='AWGN', **kwargs):
        super().__init__()
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

    def call(self, x):
        b, c, _ = x.shape
        x = tf.reshape(x, (-1, 8, 8, c*2//64))

        x = self.dec01(x)
        x = self.dec02(x)
        x = self.dec03(x)
        x = self.dec1(x)
        x = self.dec2(x)
        pred = self.dec_proj(x)

        return pred
