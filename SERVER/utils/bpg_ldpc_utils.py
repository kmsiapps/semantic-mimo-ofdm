import os

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import ebnodb2no
from sionna.channel import AWGN, FlatFadingChannel

import tensorflow as tf
from PIL import Image
import math
import numpy as np


class BPGEncoder():
    def __init__(self, working_directory='../analysis/temp'):
        '''
        working_directory: directory to save temp files
                           do not include '/' in the end
        '''
        self.working_directory = working_directory
    
    def run_bpgenc(self, qp, input_dir, output_dir='temp.bpg'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'../bpg/libbpg-0.9.8/bpgenc {input_dir} -q {qp} -o {output_dir} -f 444')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1
    
    def get_qp(self, input_dir, byte_threshold, output_dir='temp.bpg'):
        '''
        iteratively finds quality parameter that maximizes quality given the byte_threshold constraint
        '''
        # rate-match algorithm
        quality_max = 51
        quality_min = 0
        quality = (quality_max - quality_min) // 2
        
        while True:
            qp = 51 - quality
            bytes = self.run_bpgenc(qp, input_dir, output_dir)
            if quality == 0 or quality == quality_min or quality == quality_max:
                break
            elif bytes > byte_threshold and quality_min != quality - 1:
                quality_max = quality
                quality -= (quality - quality_min) // 2
            elif bytes > byte_threshold and quality_min == quality - 1:
                quality_max = quality
                quality -= 1
            elif bytes < byte_threshold and quality_max > quality:
                quality_min = quality
                quality += (quality_max - quality) // 2
            else:
                break
        
        return qp
    
    def encode(self, image_array, max_bytes, header_bytes=22):
        '''
        image_array: uint8 numpy array with shape (b, h, w, c)
        max_bytes: int, maximum bytes of the encoded image file (exlcuding header bytes)
        header_bytes: the size of BPG header bytes (to be excluded in image file size calculation)
        '''

        input_dir = f'{self.working_directory}/temp_enc.png'
        output_dir = f'{self.working_directory}/temp_enc.bpg'

        im = Image.fromarray(image_array, 'RGB')
        im.save(input_dir)

        qp = self.get_qp(input_dir, max_bytes + header_bytes, output_dir)
        
        if self.run_bpgenc(qp, input_dir, output_dir) < 0:
            raise RuntimeError("BPG encoding failed")

        # read binary and convert it to numpy binary array with float dtype
        return np.unpackbits(np.fromfile(output_dir, dtype=np.uint8)).astype(np.float32)


class BPGDecoder():
    def __init__(self, working_directory='../analysis/temp'):
        '''
        working_directory: directory to save temp files
                           do not include '/' in the end
        '''
        self.working_directory = working_directory
    
    def run_bpgdec(self, input_dir, output_dir='temp.png'):
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.system(f'../bpg/libbpg-0.9.8/bpgdec {input_dir} -o {output_dir}')

        if os.path.exists(output_dir):
            return os.path.getsize(output_dir)
        else:
            return -1

    def decode(self, bit_array, image_shape, return_flag=False):
        '''
        returns decoded result of given bit_array.
        if bit_array is not decodable, then returns the mean CIFAR-10 pixel values.

        byte_array: float array of '0' and '1'
        image_shape: used to generate image with mean pixel values if the given byte_array is not decodable
        '''
        input_dir = f'{self.working_directory}/temp_dec.bpg'
        output_dir = f'{self.working_directory}/temp_dec.png'

        byte_array = np.packbits(bit_array.astype(np.uint8))
        with open(input_dir, "wb") as binary_file:
            binary_file.write(byte_array.tobytes())

        cifar_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]) * 255
        cifar_mean = np.reshape(cifar_mean, [1] * (len(image_shape) - 1) + [3]).astype(np.uint8)

        if self.run_bpgdec(input_dir, output_dir) < 0:
            # print('warning: Decode failed. Returning mean pixel value')
            output = 0 * np.ones(image_shape) + cifar_mean
            success = False
        else:
            x = np.array(Image.open(output_dir).convert('RGB'))
            if x.shape != image_shape:
                output = 0 * np.ones(image_shape) + cifar_mean
                success = False
            else:
                output = x
                success = True
        if return_flag:
            return output, success
        else:
            return output


class LDPCTransmitter():
    '''
    Transmits given bits (float array of '0' and '1') with LDPC.
    '''
    def __init__(self, k, n, m, esno_db, channel='AWGN'):
        '''
        k: data bits per codeword (in LDPC)
        n: total codeword bits (in LDPC)
        m: modulation order (in m-QAM)
        esno_db: channel SNR
        channel: 'AWGN' or 'Rayleigh'
        '''
        self.k = k
        self.n = n
        self.num_bits_per_symbol = round(math.log2(m))

        constellation_type = 'qam' if m != 2 else 'pam'
        self.constellation = Constellation(constellation_type, num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper('app', constellation=self.constellation)
        self.channel = AWGN() if channel == 'AWGN' else FlatFadingChannel
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20)
        self.esno_db = esno_db
    

    def send(self, source_bits):
        '''
        source_bits: float np array of '0' and '1', whose total # of bits is divisible with k
        '''
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        no = ebnodb2no(self.esno_db, num_bits_per_symbol=1, coderate=1)
        c = self.encoder(u)
        x = self.mapper(c)
        y = self.channel([x, no])
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len(source_bits)]
    

    def encode(self, source_bits):
        lcm = np.lcm(self.k, self.num_bits_per_symbol)
        source_bits_pad = tf.pad(source_bits, [[0, math.ceil(len(source_bits)/lcm)*lcm - len(source_bits)]])
        u = np.reshape(source_bits_pad, (-1, self.k))

        c = self.encoder(u)
        x = self.mapper(c)

        return x


    def decode(self, y, len_source_bits):
        no = ebnodb2no(self.esno_db, num_bits_per_symbol=1, coderate=1)
        llr_ch = self.demapper([y, no])
        u_hat = self.decoder(llr_ch)

        return tf.reshape(u_hat, (-1))[:len_source_bits]


