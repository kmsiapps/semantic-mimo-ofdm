# ofdm_docker_server.py is required for the server. 
# Please move the file to the server and change the code line

# %%
import socket
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image as pilimg
import uhd
import struct
from IPython import display
import random

from usrp_utils import sendAndReceive
from utils import interleave
from ofdm_cdh import OFDM_FrameGenerator
from tcp_configs import USRP_ADDR0, USRP_ADDR1, SERVER_HOST, SERVER_PORT

# Setting initialization
total_symbols = 64 * 512
Fs = 15 * 128* 1000
Fc = 2.0e9

# Transmission Preparation
mapper = OFDM_FrameGenerator(
    num_subcarriers=72,
    DC_guard=1,
    slots_per_frame=50,
    symbols_per_slot=7,
    pilot_place=0,
    sync_place=0,
    subcarrier_spacing = 15 * 1000, 
    FFT_size = 128,
    num_cp_samples = 9,
    num_ex_cp_samples = 10,
    sequential_mapping=True,
    initial_pad = True,
    num_antenna=2,
    antenna_idx=0
)
mapper_2 = OFDM_FrameGenerator(
    num_subcarriers=72,
    DC_guard=1,
    slots_per_frame=50,
    symbols_per_slot=7,
    pilot_place=0,
    sync_place=0,
    subcarrier_spacing = 15 * 1000, 
    FFT_size = 128,
    num_cp_samples = 9,
    num_ex_cp_samples = 10,
    sequential_mapping=True,
    initial_pad = True,
    num_antenna=2,
    antenna_idx=1
)
mapper.showMap()
mapper_2.showMap()
modulation_order = 16
print(mapper.num_data+mapper_2.num_data)


#%%
usrp = uhd.usrp.MultiUSRP(f'addr0={USRP_ADDR0}, addr1={USRP_ADDR1}')

#%%
usrp.set_clock_source("external", 1)
usrp.set_time_source("external", 1)
usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
print("USRP loaded")

# Device 1: use daughterboard A, B for 0th, 1st Rx channel for both devices (0, 1)
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"), 0)  # Transmit on first device
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(""), 1)  # No transmission on first device
usrp.set_tx_antenna("TX/RX", 0)
usrp.set_tx_antenna("TX/RX", 1)

usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"), 1)  # Transmit on daughterboard A
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(""), 0)  # No transmission on first device
usrp.set_rx_antenna("TX/RX", 0)
usrp.set_rx_antenna("TX/RX", 1)

print('Tx:')
print(usrp.get_tx_subdev_spec(0))
print(usrp.get_tx_subdev_spec(1))
print('Rx:')
print(usrp.get_rx_subdev_spec(0))
print(usrp.get_rx_subdev_spec(1))

#%%
# Socket Connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_HOST, SERVER_PORT))
print("TCP Connected.")

# Useful Function for Socket Connection
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

print("Session Ready.")

#%%
def OFDM_MIMO_Session(TargetSNR, codec_type='semantic', use_interleaving=False, Tx_precode=True, use_plotting=True, k=3072, n=6144, m=4, payload_norm=4, pilot_norm=4, Tx_gain=5.5, PAPR_reduction=None):
    # Semantic model setting
    if codec_type == 'semantic':
        if PAPR_reduction not in [None, 'S', 'W']:
            print("PAPR reduction mode only can be 'S'(strong) or 'W'(weak)")
            return
        if PAPR_reduction is None:
            PAPR_reduction = ''
        mode = str(TargetSNR)+PAPR_reduction
        sock.send(f'mde {mode:<3}'.encode())
        log = full_recv(sock, 4).decode()
        if log == 'fail':
            print("No proper model for current setting")
            return

    Rx_gain=24
    Tx_chan=[0, 1]
    Rx_chan=[0, 1]
    num_Tx = len(Tx_chan)
    mappers = [mapper, mapper_2]
            
    # Image Preprocessing
    images = [
        'sample_img.png',
        '0.png',
        '1.png',
        '2.png',
        '3.png',
        '4.png',
    ]
    origin_image = np.array(pilimg.open(random.choice(images)).convert('RGB'))
    image = origin_image.reshape(8, 32, 8, 96).transpose(0, 2, 1, 3).tobytes()

    # Encode Image into Symbols
    if codec_type == 'semantic':
        sock.send('encoder'.encode())
        sock.send(image)
        payload_byte = full_recv(sock, 262144) # 64 * 8 * 8 * 16 * 4
        payload_val = np.frombuffer(payload_byte, np.float32).reshape(32768, 2)
        payload = payload_val[:,0] + 1j*payload_val[:,1]
    elif codec_type == 'bpg_ldpc':
        sock.send('bpg_enc'.encode())
        sock.send(int(k).to_bytes(length=4, byteorder='big', signed=False))
        sock.send(int(n).to_bytes(length=4, byteorder='big', signed=False))
        sock.send(int(m).to_bytes(length=4, byteorder='big', signed=False))
        sock.send(origin_image.tobytes())
        payload_byte = full_recv(sock, None) # 64 * 8 * 8 * 16 * 4
        payload = np.frombuffer(payload_byte, np.complex64).copy()

    gt = payload.copy()
    
    # ======================== MIMO channel estimation (Tx)
    if Tx_precode:
        waveforms = []
        for i in range(num_Tx):
            symbol_frame = mappers[i].mapToFrame(np.ones((64 * 256,), np.complex64))
            waveform = mappers[i].symbolsToSignal(symbol_frame).signal
            waveforms.append(waveform)
        waveform = np.stack(waveforms, axis=0)

        rcv_waveform_multi = sendAndReceive(usrp, waveform, 1, Fc, Fs, Tx_gain, Rx_gain, Tx_chan, Rx_chan,
            wait_time=0.05, tx_delay_samples=100, rx_trailing_samples=2000, otw_format='sc16')
        rcv_waveform_multi = rcv_waveform_multi[:, 10:] # initial spike removal
        rcv_waveform_multi -= np.mean(rcv_waveform_multi) # DC removal
        
        rcv_waveform_raw = np.sum(rcv_waveform_multi, axis=0)
        est_idx = mapper.synchronize(rcv_waveform_raw)

        channels = []
        for c in Rx_chan:
            rcv_waveform_raw = rcv_waveform_multi[c]
            rcv_waveform = rcv_waveform_raw[est_idx:est_idx+mapper_2.n_samples_per_frame]
            rcv_symbol = mappers[c].signalToSymbols(rcv_waveform)
            channels.append(mappers[c].get_mimo_channel(rcv_symbol))
        
        channels = np.stack(channels, axis=1) # (num_subcarriers, num_rx, num_tx, num_pilot_slots)
        channels_est_Tx = np.expand_dims(channels[..., -1], axis=-1)
        
        if use_plotting:
            plt.imshow(np.abs(channels_est_Tx[0, ..., 0]), cmap='gray')
            plt.colorbar()
            plt.show()

    # ======================== MIMO signal processing (Tx)
    payload_length = payload.shape[-1]
    # print(np.mean(np.abs(payload)**2))
    payload /= payload_norm
    if use_interleaving:
        interleaving_sequence = np.arange(payload_length)
        np.random.shuffle(interleaving_sequence)
        payload = interleave(payload, interleaving_sequence)
    payload = payload.reshape(-1, num_Tx).T # Assume size is dividable

    payload_antennawise = payload.copy()

    symbols_frame = []
    for i in range(num_Tx):
        symbol_frame = mappers[i].mapToFrame(payload[i])
        symbols_frame.append(symbol_frame)
    symbols_frame = np.stack(symbols_frame, axis=1) # (FFT_size, num_rx, num_slots)

    # Tx precoding ======================================
    waveforms = []
    if Tx_precode:
        symbols_precoded = mappers[0].zf_precode(symbols_frame, channels_est_Tx)
    else:
        symbols_precoded = symbols_frame
    
    for i in range(num_Tx):
        if Tx_precode:
            waveform = mappers[i].symbolsToSignal(symbols_precoded[i]).signal
        else:
            waveform = mappers[i].symbolsToSignal(symbols_precoded[:, i, :]).signal
        if pilot_norm != 1:
            waveform[822:960] /= 1.7
            waveform = waveform.reshape(-1, 960)
            waveform[:,:138] /= (pilot_norm / np.sqrt(len(Tx_chan)))
            waveform = waveform.flatten()
        waveforms.append(waveform)
    waveform = np.stack(waveforms, axis=0)


    # ======================== Signal Reception
    rcv_waveform_multi = sendAndReceive(usrp, waveform, 1, Fc, Fs, Tx_gain, Rx_gain, Tx_chan, Rx_chan)

    # ======================== Synchronization
    rcv_waveform_raw = np.sum(rcv_waveform_multi, axis=0)
    est_idx = mapper.synchronize(rcv_waveform_raw)

    # ======================== MIMO channel estimation (Rx)
    channels = []
    rcv_symbols = []
    for c in Rx_chan:
        rcv_waveform_raw = rcv_waveform_multi[c].copy()
        rcv_waveform = rcv_waveform_raw[est_idx:est_idx+mapper_2.n_samples_per_frame]
        if pilot_norm != 1:
            rcv_waveform[822:960] *= 1.7                          # Sync norm
            rcv_waveform = rcv_waveform.reshape(-1, 960)
            rcv_waveform[:,:138] *= (pilot_norm / np.sqrt(len(Tx_chan))) # Pilot norm
            rcv_waveform = rcv_waveform.flatten()
        rcv_symbol = mappers[c].signalToSymbols(rcv_waveform)
        channels.append(mappers[c].get_mimo_channel(rcv_symbol))
        rcv_symbols.append(rcv_symbol)

    channels = np.stack(channels, axis=1) # (num_subcarriers, num_rx, num_tx, num_pilot_slots)
    rcv_symbols = np.stack(rcv_symbols, axis=1) # (FFT_size, num_rx, num_slots)

    c_matrix = np.mean(channels, axis=-1)[0]

    # ======================== MIMO signal processing (Rx)
    rcv_symbols_zf = mappers[0].mimo_zf_equalize(rcv_symbols, channels)

    rcv_payloads = []
    for i in range(len(Rx_chan)):
        rcv_payload = mapper.extractPayloads(rcv_symbols_zf[i])
        rcv_payload = rcv_payload[:payload_length//len(Rx_chan)].astype(np.complex64)
        rcv_payloads.append(rcv_payload)
    rcv_payload_antennawise = np.stack(rcv_payloads, axis=0) 
    rcv_payload = rcv_payload_antennawise.T.flatten()

    if use_interleaving:
        rcv_payload = interleave(rcv_payload, interleaving_sequence, decode=True)
    rcv_payload *= payload_norm

    # ======================== Neural-Net decoding
    if codec_type == 'semantic':
        rcv_payload_val = np.stack([np.real(rcv_payload), np.imag(rcv_payload)], -1)
        rcv_payload_byte = rcv_payload_val.tobytes()
        sock.send('decoder'.encode())
        sock.send(rcv_payload_byte)
        rcv_image = full_recv(sock, 786432) # 64 * 32 * 32 * 3 * 4
        rcv_image = np.frombuffer(rcv_image, np.float32)
        rcv_image = rcv_image.reshape(8, 8, 32, 96).transpose(0, 2, 1, 3).reshape(256, 256, 3)
    elif codec_type == 'bpg_ldpc':
        sock.send('bpg_dec'.encode())
        rcv_payload_byte = rcv_payload.astype(np.complex64).tobytes()
        header = struct.pack('I', len(rcv_payload_byte))
        sock.send(header + rcv_payload_byte)
        rcv_image = full_recv(sock, 786432) # 64 * 32 * 32 * 3 * 4
        rcv_image = np.frombuffer(rcv_image, np.float32)
        rcv_image = rcv_image.reshape(256, 256, 3)

    # ======================== Result Calculation
    mse = np.mean((origin_image.astype(np.float32) / 255. - rcv_image)**2)
    psnr = 10*np.log10(1/mse)
    sym_pow = np.mean(np.abs(gt) ** 2)
    err_pow = np.mean(np.abs(gt - rcv_payload) ** 2)
    esnr = 10*np.log10(sym_pow/err_pow)

    if use_plotting:
        # plt.scatter(np.real(rcv_payload), np.imag(rcv_payload), s=0.1)
        # plt.scatter(np.real(gt), np.imag(gt), s=0.1)
        # plt.show()

        # plt.clf()
        
        print(np.linalg.cond(c_matrix))
        
        np.savez('symbol_antennawise.npz',
                 payload=payload_antennawise,
                 rcv_payload=rcv_payload_antennawise)
          
        colors = ['tab:blue', 'tab:orange']
        for i in range(rcv_payload_antennawise.shape[0]):
            gt = payload_antennawise[i]
            rcv_payload = rcv_payload_antennawise[i]
            sym_pow = np.mean(np.abs(gt) ** 2)
            err_pow = np.mean(np.abs(gt - rcv_payload) ** 2)

            print('gt / rcv_payload shape', gt.shape, rcv_payload.shape)

            esnr_antennawise = 10*np.log10(sym_pow/err_pow)
            print(f'esnr@ant{i}: {esnr_antennawise:.2f}dB')
            
            err = gt - rcv_payload
            err = err[:err.size//72*72].reshape(-1, 72)
            err_val = np.concatenate((np.real(err), np.imag(err)), axis=0).T
            mean_err_pow = np.mean(err_val**2, axis=1)
            mean_err_pow_dB = 10 * np.log10(mean_err_pow)
            # subcarrier_idx1 = np.arange(28, 64)
            # subcarrier_idx2 = np.arange(65, 101)
            # plt.plot(subcarrier_idx1, mean_err_pow_dB[:36], color=colors[i])
            # plt.plot(subcarrier_idx2, mean_err_pow_dB[36:], color=colors[i])
            subcarrier_idx = np.concatenate((np.arange(28, 64), np.arange(65, 101)))
            plt.plot(subcarrier_idx, mean_err_pow, label=f'Ant {i}, w/o shuffling')    
        
        if use_interleaving:
            payload_antennawise = interleave(payload_antennawise.T.flatten(), \
                interleaving_sequence, decode=True)
            payload_antennawise = payload_antennawise.reshape(-1, 2).T
            rcv_payload_antennawise = interleave(rcv_payload_antennawise.T.flatten(), \
                interleaving_sequence, decode=True)
            rcv_payload_antennawise = rcv_payload_antennawise.reshape(-1, 2).T
            for i in range(rcv_payload_antennawise.shape[0]):
                gt = payload_antennawise[i]
                rcv_payload = rcv_payload_antennawise[i]
                sym_pow = np.mean(np.abs(gt) ** 2)
                err_pow = np.mean(np.abs(gt - rcv_payload) ** 2)
                print('INTL : gt / rcv_payload shape', gt.shape, rcv_payload.shape)
                esnr_antennawise = 10*np.log10(sym_pow/err_pow)
                print(f'INTL : esnr@ant{i}: {esnr_antennawise:.2f}dB')
                
                err = gt - rcv_payload
                err = err[:err.size//72*72].reshape(-1, 72)
                err_val = np.concatenate((np.real(err), np.imag(err)), axis=0).T
                mean_err_pow = np.mean(err_val**2, axis=1)
                mean_err_pow_dB = 10 * np.log10(mean_err_pow)
                subcarrier_idx = np.concatenate((np.arange(28, 64), np.arange(65, 101)))
                plt.plot(subcarrier_idx, mean_err_pow, label=f'Ant {i}, w/ shuffling')
        plt.ylabel('Noise Power')
        plt.xlabel('Subcarrier Index')
        plt.grid()
        plt.legend()
        plt.savefig('NoisePower.png', dpi=300)
        plt.show()
        
        # plt.clf()
        # plt.figure(figsize=(8, 8))
        # for i in range(2):
        #     plt.subplot(2, 1, i+1)
        #     # waveform_wo_pilot = waveform[i].reshape(-1, 960)
        #     # waveform_wo_pilot = waveform_wo_pilot[:,138:].flatten()[137*6:]
        #     # plt.plot(np.real(waveform_wo_pilot[:200]))
        #     plt.plot(np.imag(waveform[i]))
        #     plt.title(f'Sended real signal without pilot to Tx {i}')
        # plt.show()
        
        # plt.clf()
        # plt.figure(figsize=(8, 8))
        # for i in range(2):
        #     plt.subplot(2, 1, i+1)
        #     # waveform_wo_pilot = waveform[i].reshape(-1, 960)
        #     # waveform_wo_pilot = waveform_wo_pilot[:,138:].flatten()[137*6:]
        #     # plt.plot(np.real(waveform_wo_pilot[:200]))
        #     plt.plot(np.imag(rcv_waveform_multi[i][18000:70000]))
        #     plt.title(f'Received real signal without pilot to Tx {i}')
        # plt.show()
        
        # symbol_err = rcv_payload - gt
        # symbol_snr = np.abs(symbol_err)**2 / sym_pow
        # symbol_snr_db = 10*np.log10(1/symbol_snr)
        # sns.kdeplot(symbol_snr_db)
        # plt.show()
        
        # symbol_err_val = np.concatenate([np.real(symbol_err), np.imag(symbol_err)])
        # sns.kdeplot(symbol_err_val)
        # plt.show()
        
        # plt.clf()
        # plt.figure(figsize=(12,6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(origin_image)
        # plt.title('Original Image')
        # plt.subplot(1, 2, 2)
        # plt.imshow(rcv_image)
        # plt.title('Received Image')
        # plt.suptitle(f"PSNR : {psnr:.2f}dB | ESNR : {esnr:.2f}dB")
        # plt.show()
        
    return rcv_image, psnr, esnr, channels, rcv_payload_antennawise, rcv_waveform_multi

# Virtual Session
def VirtualSession(TargetSNR, channel, codec_type='semantic', k=3072, n=6144, m=16):
    if codec_type == 'semantic':
        mode = str(TargetSNR)
        sock.send(f'mde {mode:<3}'.encode())
        log = full_recv(sock, 4).decode()
        if log == 'fail':
            print("No proper model for current setting")
            return
    
    # Image Preprocessing
    origin_image = np.array(pilimg.open(f'sample_img.png').convert('RGB'))
    image = origin_image.reshape(8, 32, 8, 96).transpose(0, 2, 1, 3).tobytes()

    # Encode Image into Symbols
    if codec_type == 'semantic':
        sock.send('encoder'.encode())
        sock.send(image)
        payload_byte = full_recv(sock, 262144) # 64 * 8 * 8 * 16 * 4
        payload_val = np.frombuffer(payload_byte, np.float32).reshape(32768, 2)
    elif codec_type == 'bpg_ldpc':
        sock.send('bpg_enc'.encode())
        sock.send(int(k).to_bytes(length=4, byteorder='big', signed=False))
        sock.send(int(n).to_bytes(length=4, byteorder='big', signed=False))
        sock.send(int(m).to_bytes(length=4, byteorder='big', signed=False))
        sock.send(origin_image.tobytes())
        payload_byte = full_recv(sock, None) # 64 * 8 * 8 * 16 * 4
        payload = np.frombuffer(payload_byte, np.complex64).copy()
        payload_val = np.stack([np.real(payload), np.imag(payload)], axis=-1)

    real_snr = 10 ** (TargetSNR / 10)
    if channel == 'AWGN':
        # AWGN
        sig_power = np.mean(np.abs(payload_val)**2)
        n = np.random.normal(loc=0, scale=(sig_power/real_snr)**0.5, size=payload_val.size).reshape(payload_val.shape).astype(np.float32)
        rcv_payload_val = payload_val + n
    elif channel == 'Rician':
        K = 100
        sigma = (1 / (2 * K))**0.5
        h_multipath = np.random.normal(loc=0, scale=sigma, size=32768*2).reshape(32768, 2).astype(np.float32)
        h = (h_multipath[:,0] + 1j * h_multipath[:,1]) + (0.5**0.5 + 1j*0.5**0.5)
        x = payload_val[:,0] + 1j * payload_val[:,1]
        hx = h * x
        sig_power = np.mean(np.abs(hx)**2)
        n = np.random.normal(loc=0, scale=(sig_power/(2*real_snr))**0.5, size=32768*2).reshape(32768, 2).astype(np.float32)
        n = n[:,0] + 1j * n[:,1]
        y = hx + n
        y /= h
        rcv_payload_val = np.stack([np.real(y), np.imag(y)], -1)

    # Decode Symbols into Image
    if codec_type == 'semantic':
        rcv_payload_byte = rcv_payload_val.tobytes()
        sock.send('decoder'.encode())
        sock.send(rcv_payload_byte)
        rcv_image = full_recv(sock, 786432) # 64 * 32 * 32 * 3 * 4
        rcv_image = np.frombuffer(rcv_image, np.float32)
        rcv_image = rcv_image.reshape(8, 8, 32, 96).transpose(0, 2, 1, 3).reshape(256, 256, 3)
    elif codec_type == 'bpg_ldpc':
        sock.send('bpg_dec'.encode())
        rcv_payload = rcv_payload_val[..., 0] + 1j*rcv_payload_val[..., 1]
        rcv_payload_byte = rcv_payload.astype(np.complex64).tobytes()
        header = struct.pack('I', len(rcv_payload_byte))
        sock.send(header + rcv_payload_byte)
        rcv_image = full_recv(sock, 786432) # 64 * 32 * 32 * 3 * 4
        rcv_image = np.frombuffer(rcv_image, np.float32)
        rcv_image = rcv_image.reshape(256, 256, 3)

    # Result Calculation
    mse = np.mean((origin_image.astype(np.float32) / 255. - rcv_image)**2)
    psnr = 10*np.log10(1/mse)
    sym_pow = np.mean(payload_val**2)
    err_pow = np.mean((payload_val - rcv_payload_val)**2)
    esnr = 10*np.log10(sym_pow/err_pow)
    
    gt = payload_val.copy()
    symbol_err_val = (rcv_payload_val - gt).flatten()
    symbol_snr = np.abs(symbol_err_val)**2 / sym_pow
    symbol_snr_db = 10*np.log10(1/symbol_snr)
    sns.kdeplot(symbol_err_val, label='simulation')
    plt.legend()
    plt.show()

    plt.clf()
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.imshow(origin_image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(rcv_image)
    plt.title('Received Image')
    plt.suptitle(f"PSNR : {psnr:.2f}dB | ESNR : {esnr:.2f}dB")
    plt.show()

    return psnr, esnr


#%%

SNR = 10

zerodb_norm_const = 28
zerodb_tx_gain = 0

norm_const = max(zerodb_norm_const / float(np.sqrt(10 ** (SNR/10))), 8.0)
tx_gain = SNR + float(10 * np.log10((norm_const / zerodb_norm_const) ** 2)) + zerodb_tx_gain
# print(norm_const)
# print(tx_gain)

N = norm_const

fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(8, 8))
gs1 = axs[0, 0].get_gridspec()
gs2 = axs[0, 2].get_gridspec()
gs3 = axs[2, 0].get_gridspec()
gs4 = axs[2, 2].get_gridspec()
for ax in axs.flatten():
    ax.remove()
ax1 = fig.add_subplot(gs1[0, 0:2])
ax2 = fig.add_subplot(gs2[0, 2:4])
ax3 = fig.add_subplot(gs3[2:, 0:2])
ax4 = fig.add_subplot(gs4[2:, 2:4])
fig.tight_layout()
dh = display.display(fig, display_id=True)

# for _ in range(50):
# while True:
rcv_image, psnr, esnr, channels, rcv_payload_antennawise, rcv_waveform_multi = OFDM_MIMO_Session(TargetSNR=SNR, codec_type='semantic', payload_norm=N, pilot_norm=4, PAPR_reduction=None, use_interleaving=True, Tx_precode=False, use_plotting=False, Tx_gain=tx_gain)

# VirtualSession(TargetSNR=SNR, codec_type='semantic', channel='AWGN')

# Channel singular value plot
channels_ = np.mean(channels, axis=-1)
d = np.linalg.svd(channels_, compute_uv=False)
ax1.clear()
ax1.plot(d)
ax1.set_xlim([0, N])
ax1.set_xlabel('Subcarrier Index')
ax1.set_ylabel('Channel Magnitude')

# ax2.psd(frame_rcv, NFFT=frame_rcv.size, Fs=sampling_rate, scale_by_freq=False)
# ax2.set_xlim([-sampling_rate//2, sampling_rate//2])
ax2.clear()
ax2.psd(rcv_waveform_multi[0], NFFT=rcv_waveform_multi[0].size, Fs=Fs, scale_by_freq=False, linewidth=0.1)
ax2.psd(rcv_waveform_multi[1], NFFT=rcv_waveform_multi[0].size, Fs=Fs, scale_by_freq=False, linewidth=0.1, alpha=0.5)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power Spectrum (dB)')

ax3.clear()
ax3.scatter(np.real(rcv_payload_antennawise[0]), np.imag(rcv_payload_antennawise[0]), s=0.1)
ax3.scatter(np.real(rcv_payload_antennawise[1]), np.imag(rcv_payload_antennawise[1]), s=0.1, alpha=0.5)
ax3.set_xlim([-.5, .5])
ax3.set_ylim([-.5, .5])
ax3.set_xlabel('In-Phase')
ax3.set_ylabel('Quadrature-Phase')
ax3.set_title('Constellations')

ax4.clear()
ax4.imshow(rcv_image)
ax4.set_title(f'ESNR:{esnr:.2f} dB, PSNR: {psnr:.2f} dB')

dh.update(fig)
plt.close()

#%%