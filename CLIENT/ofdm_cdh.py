import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from usrp_signal import USRP_signal
from utils import generateZadoffChuSymbols, generatePilotSymbols
    
class OFDM_FrameGenerator:
    def __init__(self, num_subcarriers, DC_guard, symbols_per_slot, slots_per_frame, pilot_place, sync_place,
                 subcarrier_spacing, FFT_size, num_cp_samples, num_ex_cp_samples, sequential_mapping, initial_pad,
                 num_antenna, antenna_idx):
        
        self.num_subcarriers = num_subcarriers
        self.num_guards = FFT_size - num_subcarriers - 1 - DC_guard
        self.symbols_per_slot = symbols_per_slot
        self.slots_per_frame = slots_per_frame
        self.pilot_place = pilot_place
        self.sync_place = sync_place
        self.subcarrier_spacing = subcarrier_spacing
        self.FFT_size = FFT_size
        self.sampling_rate = subcarrier_spacing * FFT_size
        self.num_cp_samples = num_cp_samples
        self.num_ex_cp_samples = num_ex_cp_samples
        self.add_cp = num_ex_cp_samples - num_cp_samples
        self.n_samples_per_frame = ((self.FFT_size + self.num_cp_samples) * self.symbols_per_slot + self.add_cp) * self.slots_per_frame
        self.num_antenna = num_antenna
        self.antenna_idx = antenna_idx
        

        # Data map excl. guard bands
        self.payload_map = np.zeros((self.num_subcarriers, symbols_per_slot*slots_per_frame))

        # Attach pilot only to selected subcarrier wrt antenna ports
        self.payload_map[:,pilot_place::symbols_per_slot] = 3 # unused
        self.payload_map[antenna_idx::num_antenna,pilot_place::symbols_per_slot] = 1
        self.payload_map[:,(sync_place+1)*symbols_per_slot-1] = 2
        if initial_pad:
            self.payload_map[:,1:symbols_per_slot-1] = 3

        # Actual data map including guard bands
        self.data_map = np.ones([self.FFT_size, symbols_per_slot*slots_per_frame]) * 3

        self.dc_start_idx = FFT_size//2 - DC_guard//2
        self.dc_end_idx = FFT_size//2 + DC_guard//2
        self.leading_guard_end_idx = self.num_guards//2
        self.trailing_guard_start_idx = FFT_size-self.num_guards//2
        # print(self.dc_start_idx, self.dc_end_idx, self.leading_guard_end_idx, self.trailing_guard_start_idx)
        self.data_map[self.leading_guard_end_idx+1:self.dc_start_idx] = self.payload_map[:self.dc_start_idx-self.leading_guard_end_idx-1]
        self.data_map[self.dc_end_idx+1:self.trailing_guard_start_idx] = self.payload_map[self.trailing_guard_start_idx-self.dc_end_idx-1:]
        
        self.sync_idx = np.where(self.data_map == 2)
        self.pilot_idx = np.where(self.data_map == 1)
        self.data_idx = np.where(self.data_map == 0)
        
        self.sync_symbols = generateZadoffChuSymbols(self.sync_idx[0].size)
        sync_frame = np.zeros_like(self.data_map, dtype=np.complex64)
        sync_frame[self.sync_idx] = self.sync_symbols
        self.sync_signal = self.symbolsToSignal(sync_frame)
        
        self.pilot = generatePilotSymbols(self.pilot_idx[0].size, QAM_order=4, seed=1030)
        self.ref_pilot_channel = np.ones_like(self.data_map, dtype=np.complex64)
        self.ref_pilot_channel[self.pilot_idx] = self.pilot
        
        self.control_iqs = np.zeros_like(self.data_map, dtype=np.complex64)
        self.control_iqs[self.sync_idx] = self.sync_symbols
        self.control_iqs[self.pilot_idx] = self.pilot
        self.num_data = np.count_nonzero(self.data_map==0)
        self.sequential_mapping = sequential_mapping
        
        
    def showMap(self):
        plt.figure(figsize=(14, 8))
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cm.viridis.N)
        img = plt.imshow(self.data_map, norm=norm, interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5], ax=plt.gca(), shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot Symbols", "Synchronization Signal", "Guard band (DC)"])
        plt.xlabel('Time Domain Symbols')
        plt.ylabel('Subcarriers')
        plt.title('OFDM frame symbol map')
        plt.show()
        
    def mapToFrame(self, data):
        if self.num_data < data.size:
            print(f"data size is too big for the frame!: {self.num_data}")
            return
        elif self.num_data > data.size:
            new_data = np.zeros(self.num_data, dtype=np.complex64)
            new_data[:data.size] = data
            data = new_data
        data = data.flatten()
        
        if self.sequential_mapping:
            symbols = self.control_iqs.copy().T
            self.reverse_idx = np.where(self.data_map.T == 0)
            symbols[self.reverse_idx] = data
            symbols = symbols.T
        else:
            symbols = self.control_iqs.copy()
            symbols[self.data_idx] = data
            
        return symbols
        
    def symbolsToSignal(self, symbols):
        signal_wo_cp = np.fft.fftshift(symbols, axes=0)
        signal_wo_cp = np.fft.ifft(signal_wo_cp, n=self.FFT_size, axis=0, norm='ortho') * np.sqrt(self.FFT_size/self.num_subcarriers)
        signal_wo_cp = np.transpose(signal_wo_cp)
        self.a = signal_wo_cp
        
        # Add CP
        signal = np.hstack([signal_wo_cp[:,-self.num_cp_samples:], signal_wo_cp])
        signal = signal.reshape([self.slots_per_frame, -1])
        signal = np.hstack([signal[:,self.FFT_size-self.add_cp:self.FFT_size], signal])
        signal = signal.flatten().astype(np.complex64)
        
        return USRP_signal(initial_symbols=signal,
                                tone_Fs=self.sampling_rate,
                                oversample_rate=1)
        
    def signalToSymbols(self, signal):
        if isinstance(signal, USRP_signal):
            signal = signal.signal
        
        # Remove CP
        signal = signal.reshape([self.slots_per_frame, -1])
        signal = signal[:,self.add_cp:]
        signal = signal.reshape([self.symbols_per_slot*self.slots_per_frame, -1])
        signal = signal[:,self.num_cp_samples:]
        
        # FFT
        signal = np.transpose(signal)
        symbols = np.fft.fft(signal, n=self.FFT_size, axis=0, norm='ortho') / np.sqrt(self.FFT_size/self.num_subcarriers)
        symbols = np.fft.fftshift(symbols, axes=0)
        return symbols
    
    def extractPayloads(self, symbols):
        if self.sequential_mapping:
            symbols = symbols.T
            data = symbols[self.reverse_idx]
            symbols = symbols.T
        else:
            data = symbols[self.data_idx]
        return data
        
    def synchronize(self, rcv_signal):
        if isinstance(rcv_signal, USRP_signal):
            rcv_signal = rcv_signal.signal
            
        corr = np.abs(scipy.signal.correlate(rcv_signal, self.sync_signal.signal, 'full'))
        est_idx = (np.argmax(corr) + 1) % self.n_samples_per_frame 

        return est_idx

    def get_mimo_channel(self, symbols):
        '''
        return: estimated mimo channel with shape [num_subcarriers, num_tx, num_pilot_slots]
        '''

        mimo_ref_pilot = np.repeat(np.reshape(self.pilot, (self.num_subcarriers//self.num_antenna, -1)), self.num_antenna, axis=0)
        pilot_rcv = np.concatenate([
            symbols[self.leading_guard_end_idx+1:self.dc_start_idx,self.pilot_place::self.symbols_per_slot],
            symbols[self.dc_end_idx+1:self.trailing_guard_start_idx,self.pilot_place::self.symbols_per_slot]
        ], axis=0)

        mimo_est_channel = pilot_rcv / mimo_ref_pilot
        mimo_est_channel = np.reshape(mimo_est_channel, (self.num_subcarriers//self.num_antenna, self.num_antenna, -1))
        intp_mimo_est_channel = np.repeat(mimo_est_channel, self.num_antenna, axis=0)

        return intp_mimo_est_channel

    def mimo_zf_equalize(self, symbols, mimo_channel):
        '''
        symbols: (FFT_size, num_rx, num_slots)
        mimo_channel: (num_subcarriers, num_rx, num_tx, num_pilot_slots)
        '''
        _, num_rx, num_tx, _ = mimo_channel.shape
        est_channel = np.ones((self.FFT_size, num_rx, num_tx, self.symbols_per_slot*self.slots_per_frame), np.complex64)

        est_channel[self.leading_guard_end_idx+1:self.dc_start_idx, ..., self.pilot_place::self.symbols_per_slot] = \
            mimo_channel[:self.dc_start_idx-self.leading_guard_end_idx-1]
        est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx, ..., self.pilot_place::self.symbols_per_slot] = \
            mimo_channel[self.trailing_guard_start_idx-self.dc_end_idx-1:] 

        # Linear interpolation
        est_channel = np.concatenate([est_channel, est_channel[...,-self.symbols_per_slot:]], axis=-1)
        # est_channel = np.concatenate([mimo_channel, mimo_channel[...,-self.symbols_per_slot:]], axis=-1)

        i, j = self.data_idx
        d = np.reshape((j % self.symbols_per_slot) / self.symbols_per_slot, (-1, 1, 1))
        ref_idx = (j // self.symbols_per_slot) * self.symbols_per_slot
        data_channel = d * est_channel[i, ..., ref_idx] + (1-d) * est_channel[i, ..., ref_idx+self.symbols_per_slot]
        
        mimo_channel_intp = np.random.normal(0., 1., est_channel[..., :-self.symbols_per_slot].shape).astype(np.complex64)
        # Due to pilot slot, singular matrix occur when np.zeros or ones()
        # So initialize it to random matrix 

        mimo_channel_intp[i, ..., j] = data_channel
        # (FFT_size, num_rx, num_tx, num_slots)

        mimo_channel_intp = np.concatenate(
            (mimo_channel_intp[self.leading_guard_end_idx+1:self.dc_start_idx],
             mimo_channel_intp[self.dc_end_idx+1:self.trailing_guard_start_idx]), axis=0)

        mimo_channel_intp = np.transpose(mimo_channel_intp, (0, 3, 1, 2))
        # (num_subcarriers, num_slots, num_rx, num_tx)

        mimo_channel_inv = np.linalg.inv(mimo_channel_intp)
        
        symbols_valid = np.concatenate(
            (symbols[self.leading_guard_end_idx+1:self.dc_start_idx],
             symbols[self.dc_end_idx+1:self.trailing_guard_start_idx]), axis=0)
        
        symbols_eq_valid = np.einsum('sitr,sri->tsi', mimo_channel_inv, symbols_valid)

        symbols_eq = np.zeros((num_rx, self.FFT_size, self.symbols_per_slot*self.slots_per_frame), np.complex64)
        symbols_eq[:, self.leading_guard_end_idx+1:self.dc_start_idx, :] = symbols_eq_valid[:, :self.dc_start_idx-self.leading_guard_end_idx-1, :]
        symbols_eq[:, self.dc_end_idx+1:self.trailing_guard_start_idx, :] = symbols_eq_valid[:, self.trailing_guard_start_idx-self.dc_end_idx-1:, :] 

        return symbols_eq

    def zf_precode(self, symbols, mimo_channel):
        '''
        symbols: (FFT_size, num_rx, num_slots)
        mimo_channel: (num_subcarriers, num_rx, num_tx, 1)
        '''
        # Do not use ZF precoding to sync signal
        sync_signal = symbols[..., (self.sync_place+1)*self.symbols_per_slot-1].T
        # (num_rx, FFT_size)

        num_subcarriers, num_rx, num_tx, _ = mimo_channel.shape
        mimo_channel = np.broadcast_to(mimo_channel, (num_subcarriers, num_rx, num_tx, self.symbols_per_slot*self.slots_per_frame))
        
        # Expand mimo_channel to FFT_size
        est_channel = np.reshape(np.identity(num_rx), (1, num_rx, num_tx, 1)) + 0.0j
        est_channel = np.broadcast_to(est_channel, (self.FFT_size, num_rx, num_tx, self.symbols_per_slot*self.slots_per_frame)).copy()

        est_channel[self.leading_guard_end_idx+1:self.dc_start_idx] = mimo_channel[:self.dc_start_idx-self.leading_guard_end_idx-1]
        est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx] = mimo_channel[self.trailing_guard_start_idx-self.dc_end_idx-1:] 

        est_channel = np.transpose(est_channel, (0, 3, 1, 2))
        # (FFT_size, num_slots, num_rx, num_tx)

        # U, D, Vh = np.linalg.svd(est_channel)

        # num_subcarriers, time_idx, num_rx = D.shape

        # # D_ = np.mean(np.mean(D, axis=0, keepdims=True), axis=1, keepdims=True)
        # # D_ /= np.sqrt(np.sum(D_**2))
        # # D_ = np.broadcast_to(D_, D.shape)

        # # # Normalize D
        # # D_wo_guardband = np.concatenate([D[:dc_idx-leading_guard_end_idx-1], D[dc_idx-leading_guard_end_idx-1:]], axis=0)
        # # norm_const = np.mean(D_wo_guardband)
        # # D[:dc_idx-leading_guard_end_idx-1] /= norm_const
        # # D[dc_idx-leading_guard_end_idx-1:] /= norm_const

        # D_matrix = np.zeros((num_subcarriers, time_idx, num_rx, num_rx))

        # # D = np.clip(D, 1/2, 1.)
        # # plt.plot(1. / D.flatten())
        # # plt.show()

        # i, j = np.diag_indices(num_rx)
        # D_matrix[:, :, i, j] = D

        # precode_matrix = np.einsum('...ij,...jk->...ik', D_matrix, Vh)
        # mimo_channel_inv = np.linalg.inv(precode_matrix)
        # mimo_channel_inv = Vh
        # mimo_channel_inv = np.linalg.inv(Vh)

        # print(np.mean(D, axis=(0, 1)))
        # norm_factor = 1 / np.mean(D, axis=(0, 1), keepdims=True)
        # norm_factor = norm_factor / np.sum(norm_factor, axis=-1)

        # mimo_channel_inv /= np.expand_dims(norm_factor, axis=-1)

        mimo_channel_inv = np.linalg.inv(est_channel)

        norm_factor = np.sqrt(np.sum(mimo_channel_inv ** 2, axis=(-1, -2), keepdims=True))
        mimo_channel_inv /= norm_factor
        
        symbols_precoded = np.einsum('sitr,sri->tsi', mimo_channel_inv, symbols)
        symbols_precoded[..., (self.sync_place+1)*self.symbols_per_slot-1] = sync_signal

        return symbols_precoded

    def equalize(self, symbols):
        # TODO : Now hardcoded for pilot_place = 0
        est_channel = symbols / self.ref_pilot_channel
        
        # Nearest neighbor interpolation
        # Data map excl. guard bands
        payload_map = np.ones((self.num_subcarriers, self.symbols_per_slot*self.slots_per_frame), np.complex64)
        
        payload_map[:self.dc_start_idx-self.leading_guard_end_idx-1] = est_channel[self.leading_guard_end_idx+1:self.dc_start_idx]
        payload_map[self.trailing_guard_start_idx-self.dc_end_idx-1:] = est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx]
        
        # Valid pilot
        valid_pilot = payload_map[self.antenna_idx::self.num_antenna,self.pilot_place::self.symbols_per_slot]

        intp_pilot = np.repeat(valid_pilot, self.num_antenna, axis=0)

        payload_map[:,self.pilot_place::self.symbols_per_slot] = intp_pilot
        # intp value comprehended as (antenna_idx, val, num_antenna-antenna_idx-1)-sized subcarrier sequence

        # Re assign into est_channel
        
        est_channel[self.leading_guard_end_idx+1:self.dc_start_idx] = payload_map[:self.dc_start_idx-self.leading_guard_end_idx-1]
        est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx] = payload_map[self.trailing_guard_start_idx-self.dc_end_idx-1:]


        # Interpolation (time idx)
        est_channel = np.hstack([est_channel, est_channel[:,-self.symbols_per_slot:]]) #TODO

        i, j = self.data_idx
        d = (j % self.symbols_per_slot) / self.symbols_per_slot
        ref_idx = (j // self.symbols_per_slot) * self.symbols_per_slot
        data_channel = d * est_channel[i, ref_idx] + (1-d) * est_channel[i, ref_idx+self.symbols_per_slot]
        
        interpolated_channel = np.ones_like(self.data_map, dtype=np.complex64)
        interpolated_channel[i, j] = data_channel
        symbols /= interpolated_channel
        
        return symbols

    def get_channel(self, symbols):
        est_channel = symbols / self.ref_pilot_channel
        est_channel = np.hstack([est_channel, est_channel[:,-self.symbols_per_slot:]]) #TODO
        
        # avg_noise_pwr = np.var(est_channel - self.ref_pilot_channel, axis=-1)

        return np.mean(est_channel)
