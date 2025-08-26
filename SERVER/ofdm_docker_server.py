#%%
import socket
import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
import struct
import matplotlib.pyplot as plt

from config import config
from models.cvc_block import SemViT_Encoder_Only, SemViT_Decoder_Only
from utils.bpg_ldpc_utils import LDPCTransmitter, BPGEncoder, BPGDecoder

model_ckpts = {
    '0':'adap_param/train0',
    '5':'adap_param/train5',
    '10':'adap_param/train10',
    '15':'adap_param/train15',
    '20':'adap_param/train20',
    '10S':'logs/SemViT_Pp2048/weights/epoch_199',
    '10W':'logs/SemViT_Pp16384/weights/epoch_172',
    '20W':'logs/SemViT_20_Pp32768/weights/epoch_197',
    '20S':'logs/SemViT_20_Pp4096/weights/epoch_199'
}

symbol_norms = {  '0':2.524683500037204,
                  '5':2.729240523253814,
                 '10':2.9295657526786423,
                '10W':3.0270942437504114,
                '10S':3.5949122664101325,
                 '15':2.900232792422729,
                 '20':2.9444245231802224,
                '20W':3.159407443248821,
                '20S':5.05950405811547}


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv_address = ('0.0.0.0', config.PORT)
sock.bind(recv_address)
print("Listening")
sock.listen(1)
conn, addr = sock.accept()
print("TCP connected")

encoder_network = SemViT_Encoder_Only()
decoder_network = SemViT_Decoder_Only()
mode = '10'
encoder_network.load_weights(model_ckpts[mode]).expect_partial()
decoder_network.load_weights(model_ckpts[mode]).expect_partial()

bpgencoder = BPGEncoder(working_directory='./temp')
bpgdecoder = BPGDecoder(working_directory='./temp')

ldpctransmitter = None
img = None

def full_recv(sock, recv_len):
    BUFF_SIZE = 94900
    data = b''

    if recv_len is None:
        partial_data = sock.recv(4)
        recv_len, *_ = struct.unpack(
            'I',
            partial_data[:4]
        )
        data += partial_data[4:]

    with tqdm(initial=len(data), total=recv_len, leave=False) as progress:
        while len(data) < recv_len:
            partial_data = sock.recv(min(BUFF_SIZE, recv_len - len(data)))
            data += partial_data
            progress.update(len(partial_data))
    return data
  
# try:
while True:
    abi = conn.recv(7)
    print(abi)
    func = abi.decode()
    if func == 'encoder':
        img_byte = full_recv(conn, 196608) # 64 * 32 * 32 * 3  
        img = np.frombuffer(img_byte, np.uint8)
        img = img.astype(np.float32).reshape(64, 32, 32, 3) / 255.
        symbol = encoder_network(img)
        symbol /= symbol_norms[mode]
        symbol_byte = np.array(symbol).tobytes()
        conn.send(symbol_byte)
        # print(f"Symbol amp mean : {np.mean(np.array(symbol)**2)**0.5}")
    elif func == 'decoder':
        symbol_byte = full_recv(conn, 262144) # 64 * 8 * 8 * 16 * 4
        symbol = np.frombuffer(symbol_byte, np.float32)
        symbol = symbol*symbol_norms[mode]
        symbol = symbol.reshape(64, 512, 2)
        img = decoder_network(symbol)
        img_byte = np.array(img).tobytes()
        conn.send(img_byte)
    elif 'mde' in func:
        new_mode = func.split(' ')[1]
        if new_mode in model_ckpts.keys():
            if new_mode != mode:
                mode = new_mode
                encoder_network.load_weights(model_ckpts[mode]).expect_partial()
                decoder_network.load_weights(model_ckpts[mode]).expect_partial()
            conn.send('succ'.encode())
        else:
            conn.send('fail'.encode())
    elif func == 'bpg_enc':
        snr = mode.strip('W').strip('S')
        
        b = 64
        bw = 1/6

        k = int.from_bytes(conn.recv(4), byteorder='big', signed=False)
        n = int.from_bytes(conn.recv(4), byteorder='big', signed=False)
        m = int.from_bytes(conn.recv(4), byteorder='big', signed=False)
        
        max_bytes = b * 32 * 32 * 3 * bw * math.log2(m) * k / n / 8
        ldpctransmitter = LDPCTransmitter(k, n, m, float(snr))

        img_byte = full_recv(conn, 196608) # 64 * 32 * 32 * 3  
        img = np.frombuffer(img_byte, np.uint8)
        img = img.reshape(8*32, 8*32, 3)

        # Encode image
        src_bits = bpgencoder.encode(img, max_bytes)
        iq = ldpctransmitter.encode(src_bits)
        ldpc_iq_shape = iq.shape
        print('ldpc iq shape', ldpc_iq_shape)

        symbol_byte = iq.numpy().astype(np.complex64).tobytes()
        header = struct.pack('I', len(symbol_byte))
        conn.send(header + symbol_byte)
    elif func == 'bpg_dec':
        rcv_iq = full_recv(conn, None) # Dynamically receive bytes (see full_recv)
        rcv_iq = np.frombuffer(rcv_iq, np.complex64)
        rcv_iq = np.reshape(rcv_iq, (-1, n//int(math.log2(m))))

        rcv_bits = ldpctransmitter.decode(rcv_iq, len(src_bits))
        decoded_image, succ = bpgdecoder.decode(rcv_bits.numpy(), img.shape, return_flag=True)

        print(f"k:{k}, k:{n}, k:{m}, succ:{succ}, ")

        decoded_image = decoded_image.astype(np.float32) / 255.0

        decoded_image = decoded_image.tobytes()
        conn.send(decoded_image)

            
# except Exception as e:
#     print('Error:', e)
#     conn.close()

# %%


