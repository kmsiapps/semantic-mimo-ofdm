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
from skimage.metrics import structural_similarity

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
# usrp = uhd.usrp.MultiUSRP(f'addr0={USRP_ADDR0}')
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
    origin_image = np.array(pilimg.open(f'sample_img.png').convert('RGB'))
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

        rcv_waveform_multi = sendAndReceive(usrp, waveform, 1, Fc, Fs, Tx_gain, Rx_gain, Tx_chan, Rx_chan)
        
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
    
    R_ssim = structural_similarity(origin_image[:,:,0], (rcv_image*255.).astype(np.uint8)[:,:,0])
    G_ssim = structural_similarity(origin_image[:,:,1], (rcv_image*255.).astype(np.uint8)[:,:,1])
    B_ssim = structural_similarity(origin_image[:,:,2], (rcv_image*255.).astype(np.uint8)[:,:,2])
    ssim = np.mean((R_ssim, G_ssim, B_ssim))
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
        
        plt.clf()
        plt.figure(figsize=(8, 8))
        for i in range(2):
            plt.subplot(2, 1, i+1)
            # waveform_wo_pilot = waveform[i].reshape(-1, 960)
            # waveform_wo_pilot = waveform_wo_pilot[:,138:].flatten()[137*6:]
            # plt.plot(np.real(waveform_wo_pilot[:200]))
            plt.plot(np.imag(waveform[i]))
            plt.title(f'Sended real signal without pilot to Tx {i}')
        plt.show()
        
        plt.clf()
        plt.figure(figsize=(8, 8))
        for i in range(2):
            plt.subplot(2, 1, i+1)
            # waveform_wo_pilot = waveform[i].reshape(-1, 960)
            # waveform_wo_pilot = waveform_wo_pilot[:,138:].flatten()[137*6:]
            # plt.plot(np.real(waveform_wo_pilot[:200]))
            plt.plot(np.imag(rcv_waveform_multi[i][18000:70000]))
            plt.title(f'Received real signal without pilot to Tx {i}')
        plt.show()
        
        symbol_err = rcv_payload - gt
        symbol_snr = np.abs(symbol_err)**2 / sym_pow
        symbol_snr_db = 10*np.log10(1/symbol_snr)
        sns.kdeplot(symbol_snr_db)
        plt.show()
        
        symbol_err_val = np.concatenate([np.real(symbol_err), np.imag(symbol_err)])
        sns.kdeplot(symbol_err_val)
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
        
    return psnr, esnr, ssim

# Virtual Session
def VirtualSession(TargetSNR, channel, codec_type='semantic', k=3072, n=6144, m=16, use_plotting=True):
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
    R_ssim = structural_similarity(origin_image[:,:,0], (rcv_image*255.).astype(np.uint8)[:,:,0])
    G_ssim = structural_similarity(origin_image[:,:,1], (rcv_image*255.).astype(np.uint8)[:,:,1])
    B_ssim = structural_similarity(origin_image[:,:,2], (rcv_image*255.).astype(np.uint8)[:,:,2])
    ssim = np.mean((R_ssim, G_ssim, B_ssim))
    psnr = 10*np.log10(1/mse)
    sym_pow = np.mean(payload_val**2)
    err_pow = np.mean((payload_val - rcv_payload_val)**2)
    esnr = 10*np.log10(sym_pow/err_pow)
    
    if use_plotting:
        gt = payload_val.copy()
        symbol_err_val = (rcv_payload_val - gt).flatten()
        symbol_snr = np.abs(symbol_err_val)**2 / sym_pow
        symbol_snr_db = 10*np.log10(1/symbol_snr)
        sns.kdeplot(symbol_err_val, label='simulation')
        plt.legend()
        plt.show()

    return psnr, esnr, ssim


#%%

SNR = 0
INTL = False

# zerodb_norm_const = 145
# zerodb_tx_gain = 0

# norm_const = max(zerodb_norm_const / float(np.sqrt(10 ** (SNR/10))), 8.0)
# tx_gain = SNR + float(10 * np.log10((norm_const / zerodb_norm_const) ** 2)) + zerodb_tx_gain
# print(norm_const)
# print(tx_gain)

N_vals = {20:25, 15:46, 10:85, 5:150, 0:280} # (semantic)
N_vals = {k:v/1.414 for k, v in N_vals.items()} # bpg-ldpc
Tx_gain = {20:0, 15:0, 10:0, 5:0, 0:0}

N = N_vals[SNR]
tx_gain = Tx_gain[SNR]

k = {
    0: 3072,
    5: 3072,
    10: 3072,
    15: 3072,
    20: 3072
}

n = {
    0: 4608,
    5: 4608,
    10: 4608,
    15: 4608,
    20: 4608
}

m = {
    0: 2,
    5: 4,
    10: 16,
    15: 64,
    20: 256
}

# N = 12.3
# tx_gain = 10
# N_vals_fix = {20:4.2, 15:9.7, 10:18.1, 5:33.1, 0:62}
# N_vals_move = {20:4.2, 15:7.1, 10:12.3, 5:18.2, 0:29}

# OFDM_MIMO_Session(TargetSNR=SNR, codec_type='semantic', payload_norm=N, pilot_norm=4, PAPR_reduction=None, use_interleaving=INTL, Tx_precode=False, use_plotting=True, Tx_gain=tx_gain)
OFDM_MIMO_Session(
    TargetSNR=SNR, codec_type='bpg_ldpc', payload_norm=N, pilot_norm=4, PAPR_reduction=None, use_interleaving=INTL, Tx_precode=False, use_plotting=True, Tx_gain=tx_gain,
    k=k[SNR],
    n=n[SNR],
    m=m[SNR]
)

# VirtualSession(TargetSNR=SNR, codec_type='semantic', channel='AWGN')

# psnr, esnr, ssim = VirtualSession(
#     TargetSNR=SNR,
#     codec_type='bpg_ldpc',
#     channel='AWGN',
#     use_plotting=False
# )

# For digital gain control, use
# 38 / float(np.sqrt(10 ** (SNR/10)))
# Note that in 15, 20 dB, it doesn't scale linearly

# With PA,
# Norm_const = 30 for 0 dB SNR, 8V voltage

# Why Tx_gain doesn't increase the ESNR?
# from Tx gain with 15 or 20, nonlinear region...


#%% Interleaving

num_repetition = 10
TargetSNRs = [0, 5, 10, 15, 20]

experiments = {}
total_num_experiments = len(TargetSNRs) * 3 * num_repetition
t = tqdm(total=total_num_experiments)


N_vals = {20:22, 15:25, 10:45, 5:80, 0:140}
Tx_gain = {20:5, 15:0, 10:0, 5:0, 0:0}

for use_interleaving in [True, False]:
    exp_name = f'INTL={use_interleaving}'
    experiments[exp_name] = []
    for snr in TargetSNRs:
        for i in range(num_repetition):
            SNR = snr
            
            N = N_vals[SNR]
            tx_gain = Tx_gain[SNR]
            
            psnr, esnr, ssim = OFDM_MIMO_Session(
                TargetSNR=snr, 
                codec_type='semantic', 
                payload_norm=N, 
                pilot_norm=4, 
                PAPR_reduction=None, 
                use_interleaving=use_interleaving, 
                Tx_precode=False, 
                use_plotting=False, 
                Tx_gain=tx_gain
            )
            if esnr < -20:
                print(f'DISCARD: INTL:{use_interleaving}, SNR:{snr}, No:{i}')
                continue
            experiments[exp_name].append((esnr, psnr, ssim))
            t.update(1)

experiments['awgn'] = []
for snr in TargetSNRs:
    for i in range(num_repetition):
        psnr, esnr, ssim = VirtualSession(TargetSNR=snr, codec_type='semantic', channel='AWGN', use_plotting=False)
        experiments['awgn'].append((esnr, psnr, ssim))
        t.update(1)
t.close()

np.savez('result_semantic_250423.npz', 
         **experiments
         )


#%% Normal TEST

num_repetition = 5
TargetSNRs = [0, 5, 10, 15, 20]

experiments = {}
total_num_experiments = len(TargetSNRs) * 5 * num_repetition
t = tqdm(total=total_num_experiments)

N_vals_fix = {20:4.4, 15:11.7, 10:22.2, 5:38.5, 0:69.}
N_vals_move = {20:4.4, 15:8.1, 10:13., 5:21.1, 0:32.}

for pilot_gain in ['Fixed']:
    for use_interleaving in [True, False]:
        exp_name = f'pilot={pilot_gain}-INTL={use_interleaving}'
        experiments[exp_name] = []
        for snr in TargetSNRs:
            for i in range(num_repetition):
                SNR = snr
                
                zerodb_norm_const = 34
                zerodb_tx_gain = 0

                norm_const = max(zerodb_norm_const / float(np.sqrt(10 ** (SNR/10))), 8.0)
                tx_gain = SNR + float(10 * np.log10((norm_const / zerodb_norm_const) ** 2)) + zerodb_tx_gain
                
                N = norm_const
                
                if pilot_gain == 'Fixed':
                    # N = N_vals_fix[snr]
                    pN = 4
                else:
                    # N = N_vals_move[snr]
                    pN = N
                
                psnr, esnr = OFDM_MIMO_Session(
                    TargetSNR=snr, 
                    codec_type='semantic', 
                    payload_norm=N, 
                    pilot_norm=pN, 
                    PAPR_reduction=None, 
                    use_interleaving=use_interleaving, 
                    Tx_precode=False, 
                    use_plotting=False, 
                    Tx_gain=tx_gain
                )
                experiments[exp_name].append((esnr, psnr))
                t.update(1)

experiments['awgn'] = []
for snr in TargetSNRs:
    for i in range(num_repetition):
        psnr, esnr = VirtualSession(TargetSNR=snr, codec_type='semantic', channel='AWGN')
        experiments['awgn'].append((esnr, psnr))
        t.update(1)
t.close()

np.savez('result_semantic.npz', 
         **experiments
         )

#%%

result = np.load('result_semantic_250423.npz')
keys = list(result.keys())
s = 7


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for idx, key in enumerate(keys):
    plt.scatter(result[key][..., 0], result[key][..., 1], s=s, alpha=0.3, color=colors[idx], label='_nolegend_')
    mean_result = np.mean(np.reshape(result[key], (-1, 5, 3)), axis=1)
    plt.title("PSNR")
    plt.plot(mean_result[..., 0], mean_result[..., 1], 'x-', alpha=1.0, color=colors[idx])

plt.xlabel('Effective SNR (dB)')
plt.ylabel('PSNR (dB)')
plt.title('ESNR/PSNR result')
plt.legend(keys)
plt.savefig('rlt.png', dpi=300)
plt.show()


for idx, key in enumerate(keys):
    plt.scatter(result[key][..., 0], result[key][..., 2], s=s, alpha=0.3, color=colors[idx], label='_nolegend_')
    mean_result = np.mean(np.reshape(result[key], (-1, 5, 3)), axis=1)
    plt.title("PSNR")
    plt.plot(mean_result[..., 0], mean_result[..., 2], 'x-', alpha=1.0, color=colors[idx])

plt.xlabel('Effective SNR (dB)')
plt.ylabel('SSIM')
plt.title('ESNR/SSIM result')
plt.legend(keys)
plt.savefig('rlt.png', dpi=300)
plt.show()


#%% PAPR TEST

num_repetition = 10

experiments = {}
t = tqdm(total=3*num_repetition)
exps = ['none', 'weak', 'strong']
for exp in exps:
    experiments[exp] = []

for i in range(num_repetition):
    for papr, exp in zip([None, 'W', 'S'], exps):
        SNR = 20
        
        zerodb_norm_const = 34
        zerodb_tx_gain = 0

        norm_const = max(zerodb_norm_const / float(np.sqrt(10 ** (SNR/10))), 8.0)
        tx_gain = SNR + float(10 * np.log10((norm_const / zerodb_norm_const) ** 2)) + zerodb_tx_gain

        psnr, esnr = OFDM_MIMO_Session(
            TargetSNR=20, 
            codec_type='semantic', 
            payload_norm=N, 
            pilot_norm=N, 
            PAPR_reduction=papr, 
            use_interleaving=False, 
            Tx_precode=False, 
            use_plotting=False, 
            Tx_gain=8.
        )
        experiments[exp].append((esnr, psnr))
        t.update(1)
t.close()

# experiments['awgn'] = []
# for snr in TargetSNRs:
#     for i in range(num_repetition):
#         psnr, esnr = VirtualSession(TargetSNR=snr, codec_type='semantic', channel='AWGN')
#         experiments['awgn'].append((esnr, psnr))
#         t.update(1)
# t.close()

np.savez('result_semantic_papr.npz', 
         **experiments
         )

result = np.load('result_semantic_papr.npz')
keys = list(result.keys())
s = 7

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for idx, key in enumerate(keys):
    plt.scatter(result[key][..., 0], result[key][..., 1], s=s, alpha=0.3, color=colors[idx], label='_nolegend_')
    mean_result = np.mean(np.reshape(result[key], (1, -1, 2)), axis=1)
    plt.plot(mean_result[..., 0], mean_result[..., 1], 'D-', alpha=1.0, color=colors[idx])

plt.xlabel('Effective SNR (dB)')
plt.ylabel('PSNR (dB)')
plt.title('ESNR/PSNR result')
plt.legend(keys)
plt.savefig('rlt.png', dpi=300)
plt.show()

#%% Find Good Norm_const

num_repetition = 10
# norm_consts = np.arange(10, 25, 1)
norm_consts_ref = 3
norm_consts_rel_db = np.arange(0, 22, 1)
norm_consts_rel = 10 ** (norm_consts_rel_db / 20) # Since norm_consts are amplitude, use 20
norm_consts = norm_consts_ref * norm_consts_rel

experiments = {}
t = tqdm(total=2*num_repetition*len(norm_consts))
exps = ['none', 'weak', 'strong']
for exp in exps:
    experiments[exp] = []


for N in norm_consts:
    for papr, exp in zip([None, 'W', 'S'], exps):
        for i in range(num_repetition):
            psnr, esnr = OFDM_MIMO_Session(
                TargetSNR=20, 
                codec_type='semantic', 
                payload_norm=N, 
                pilot_norm=N, 
                PAPR_reduction=papr, 
                use_interleaving=False, 
                Tx_precode=False, 
                use_plotting=False, 
                Tx_gain=8.
            )
            experiments[exp].append((N, esnr, psnr))
            t.update(1)
t.close()

np.savez('N_ENSR_PSNR_movePilot_dB.npz', 
         **experiments
         )

result = np.load('N_ENSR_PSNR_movePilot_dB.npz')
keys = list(result.keys())
s = 3

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for idx, key in enumerate(keys):
    plt.scatter(result[key][..., 0], result[key][..., 1], s=s, alpha=0.3, color=colors[idx], label='_nolegend_')
    mean_result = np.mean(np.reshape(result[key], (-1, 10, 3)), axis=1)
    plt.plot(mean_result[..., 0], mean_result[..., 1], 'D-', markersize=s, alpha=1.0, color=colors[idx])

plt.xlabel('Norm const')
plt.ylabel('ESNR (dB)')
plt.title('N/ESNR result')
plt.legend(keys)
plt.savefig('rlt.png', dpi=300)
plt.show()

#%% Plot above experiment prittier

result = np.load('N_ENSR_PSNR_movePilot_dB.npz')
keys = list(result.keys())
s = 3
plt.figure(figsize=(8, 6))

# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
colors = ['mediumorchid', 'c', '#FF7900']
captions = ['Baseline', 'Weak PAPR Reduction', 'Strong PAPR Reduction']

Tx_dB_ref = -20 * np.log10(1 / 3) + 8 # + 28.6
for idx, key in enumerate(keys):
    mean_result = np.mean(np.reshape(result[key], (-1, 10, 3)), axis=1)
    Tx_dB = 10 * np.log10(1 / mean_result[..., 0] ** 2) + Tx_dB_ref
    # print(33 < Tx_dB)
    # note_idx = np.all([33 < Tx_dB, Tx_dB < 33.2])
    # note_idx = np.where(note_idx)
    print(mean_result[4, 1])
    plt.plot(Tx_dB[:-2], mean_result[2:, 2], 'D-', markersize=s, alpha=1.0, color=colors[idx])

# plt.xlim([25, 37])
# plt.ylim([28, 37.3])
# plt.xticks(np.arange(25, 37.1, 2))
# plt.yticks(np.arange(29, 37.6, 2))
plt.xlabel('PA Input, Baseband Power [dBm]')
plt.ylabel('PSNR [dB]')
# plt.title('N/ESNR result')
plt.grid()
plt.legend(captions)
plt.savefig('rlt.png', dpi=300)
plt.show()


#%%
experiments = {}

num_repetition = 3
TargetSNRs = [0, 5, 10, 15, 20]
mcs = [(k, n, m) for k, n in [(3072, 6144), (3072, 4608), (1536, 4608)] for m in (2, 4, 16, 64, 256)]
# mcs = [(4, 3072, 6144)]

total_num_experiments = len(TargetSNRs) * (len(mcs) * 2 + 1) * num_repetition
t = tqdm(total=total_num_experiments)

N_vals = {20:22, 15:25, 10:45, 5:83, 0:145}
Tx_gain = {20:5, 15:0, 10:0, 5:0, 0:0}

for k, n, m in mcs:
    for Tx_precode in [False]:
        for use_interleaving in [True, False]:
            exp_name = f'precode={Tx_precode}_interleave={use_interleaving}_k={k}_n={n}_m={m}'
            experiments[exp_name] = []
            for snr in TargetSNRs:
                N = N_vals[snr]
                tx_gain = Tx_gain[snr]
                
                for i in range(num_repetition):
                    psnr, esnr, ssim = OFDM_MIMO_Session(
                        TargetSNR=snr,
                        codec_type='bpg_ldpc',
                        Tx_precode=Tx_precode,
                        use_interleaving=use_interleaving,
                        use_plotting=False,
                        k=k,
                        n=n,
                        m=m,
                        payload_norm=N, 
                        pilot_norm=4,
                        Tx_gain=tx_gain
                    )
                    t.update(1)
                    if esnr < -20:
                        print(f'DISCARD: mcs:({k}, {n}), {m} QAM, INTL:{use_interleaving}, SNR:{snr}, No:{i}')
                        continue
                    experiments[exp_name].append((esnr, psnr, ssim))

experiments['awgn'] = []
for snr in TargetSNRs:
    for i in range(num_repetition):
        psnr, esnr, ssim = VirtualSession(TargetSNR=snr, codec_type='bpg_ldpc', channel='AWGN')
        experiments['awgn'].append((esnr, psnr, ssim))
        t.update(1)
t.close()

np.savez('result_bpg_250423.npz', 
         **experiments
         )

result = np.load('result_bpg_250423.npz')
keys = list(result.keys())
s = 7

for key in keys:
    plt.scatter(result[key][..., 0], result[key][..., 1], s=s)
plt.xlabel('Effective SNR (dB)')
plt.ylabel('PSNR (dB)')
plt.title('ESNR/PSNR result')
plt.legend(keys)
plt.savefig('rlt.png', dpi=300)

#%%

# result = np.load('./result_identical_pwr/result_semantic.npz')
# keys = list(result.keys())
# s = 7

# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# for idx, key in enumerate(keys):
#     plt.scatter(result[key][..., 0], result[key][..., 1], s=s, alpha=0.3, color=colors[idx], label='_nolegend_')
#     mean_result = np.mean(np.reshape(result[key], (len(TargetSNRs), -1, 2)), axis=1)
#     plt.plot(mean_result[..., 0], mean_result[..., 1], 'x-', alpha=1.0, color=colors[idx])

# plt.xlabel('Effective SNR (dB)')
# plt.ylabel('PSNR (dB)')
# plt.title('ESNR/PSNR result')
# plt.legend(keys)
# plt.savefig('rlt.png', dpi=300)

result = np.load('./result_semantic.npz')
keys = list(result.keys())
s = 7

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for idx, key in enumerate(keys):
    
    mean_result = np.mean(np.reshape(result[key], (len(TargetSNRs), -1, 2)), axis=1)
    plt.plot(mean_result[..., 0], mean_result[..., 1], 'x-', alpha=1.0, color=colors[idx])

plt.xlabel('Effective SNR (dB)')
plt.ylabel('PSNR (dB)')
plt.title('ESNR/PSNR result')
plt.legend(keys)
plt.savefig('rlt.png', dpi=300)
plt.show()

#%%

result = np.load('result_bpg.npz')
keys = list(result.keys())
s = 7

for key in keys:
    plt.scatter(result[key][..., 0], result[key][..., 1], s=s)
plt.xlabel('Effective SNR (dB)')
plt.ylabel('PSNR (dB)')
plt.title('ESNR/PSNR result')
# plt.legend(keys)
plt.savefig('rlt.png', dpi=300)
plt.show()

#%%
import numpy as np

a = np.random.random([72])
ac = np.concatenate([np.zeros(28), a, np.zeros(28)])
b = np.fft.ifft(a, norm='ortho')
bc = np.fft.ifft(ac, norm='ortho')
a_ = np.mean(np.abs(a)**2)
b_ = np.mean(np.abs(b)**2)
bc_ = np.mean(np.abs(bc)**2)

print(a_)
print(b_)
print(bc_)

print(bc_/b_)


# %%
