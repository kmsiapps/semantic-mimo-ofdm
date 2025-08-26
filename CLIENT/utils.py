import numpy as np
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from IPython import display
from modulate import modulate
from usrp_signal import USRP_signal

class transmissionPlot:
    def __init__(self, sampling_rate=None, dynamic=True):
        self.fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(8, 10))
        gs1 = axs[0, 0].get_gridspec()
        gs2 = axs[0, 2].get_gridspec()
        gs3 = axs[1, 0].get_gridspec()
        for ax in axs.flatten():
            ax.remove()
        self.ax1 = self.fig.add_subplot(gs1[0, 0:2])
        self.ax2 = self.fig.add_subplot(gs2[0, 2:4])
        self.ax3 = self.fig.add_subplot(gs3[1:, :])
        self.fig.tight_layout()

        self.dynamic = dynamic
        if dynamic:
            self.dh = display.display(self.fig, display_id=True)
        self.Fs = sampling_rate
        
        mplstyle.use('fast')
        
    def plot(self, signal, rx_sym, tx_sym=None):
        if isinstance(signal, USRP_signal):
            Fs = signal.tone_Fs*signal.taps_per_symbol
            signal = signal.signal
        else:
            Fs = self.Fs
        
        time = np.arange(signal.size) / Fs
        self.ax1.clear()
        self.ax1.plot(time, np.real(signal), label='Real')
        self.ax1.plot(time, np.imag(signal), label='Imag')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.legend()
        
        self.ax2.clear()
        spectrum = np.fft.fftshift(np.fft.fft(signal))
        spectrum_dB = 20 * np.log10(np.abs(spectrum))
        freq = np.fft.fftfreq(signal.size, 1/Fs)
        self.ax2.plot(np.fft.fftshift(freq), spectrum_dB)
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Power Spectrum (dB)')
        
        self.ax3.clear()
        if tx_sym is not None:
            self.ax3.scatter(np.real(tx_sym), np.imag(tx_sym), label='Transmitted')
        self.ax3.scatter(np.real(rx_sym), np.imag(rx_sym), label='Received')
        self.ax3.set_xlabel('In-Phase')
        self.ax3.set_ylabel('Quadrature-Phase')
        self.ax3.legend()
                
        if self.dynamic:
            self.dh.update(self.fig)


def generateSineSymbols(length):
    '''
    Generates Sine Wave
    '''
    seq = np.arange(length)
    sine = np.sin(2 * np.pi * seq / length * 4)
    sine = sine.astype(np.complex64)
    return sine

def generateRandomSymbols(QAM_order, num_symbols):
    '''
    Generates Random Symbols in QAM modulation.
    argument "QAM_order" should be power of 2.
    '''
    assert (QAM_order&(QAM_order-1))==0, "QAM order should be power of 2"
    
    qam = QAMModem(QAM_order)
    bits = np.random.randint(0, 2, (num_symbols)*qam.num_bits_symbol)
    symbols = qam.modulate(bits)
        
    # Normalization to power = 1
    symbols /= np.sqrt(np.mean(np.real(symbols)**2 + np.imag(symbols)**2))
    return symbols

def generateRandomSymbols_lite(QAM_order, num_symbols):
    '''
    Generates Random Symbols in QAM modulation.
    argument "QAM_order" should be power of 2.
    '''
    assert (QAM_order&(QAM_order-1))==0, "QAM order should be power of 2"
    
    bits = np.random.randint(0, 2, (num_symbols)*int(np.log2(QAM_order)))
    symbols = modulate(bits, QAM_order)
        
    # Normalization to power = 1
    symbols /= np.sqrt(np.mean(np.real(symbols)**2 + np.imag(symbols)**2))
    return symbols

def generatePilotSymbols(num_symbols, QAM_order=4, seed=0):
    '''
    Generates Pseudo-Random Symbols in QAM modulation for pilot.
    argument "QAM_order" should be power of 2.
    '''
    assert (QAM_order&(QAM_order-1))==0, "QAM order should be power of 2"
    
    qam = QAMModem(QAM_order)
    rng = np.random.RandomState(seed)
    bits = rng.randint(0, 2, (num_symbols)*qam.num_bits_symbol)
    symbols = qam.modulate(bits)
        
    # Normalization to power = 1
    symbols /= np.sqrt(np.mean(np.real(symbols)**2 + np.imag(symbols)**2))
    return symbols

def generateFixedSymbols(QAM_order):
    '''
    Generates Fixed Symbols in QAM modulation.
    argument "QAM_order" should be power of 2.
    '''
    assert (QAM_order&(QAM_order-1))==0, "QAM order should be power of 2"
    
    nbits = int(np.log2(QAM_order))
    qam = QAMModem(QAM_order)
    nums = np.unpackbits(np.arange(QAM_order).astype(np.uint8))
    bits = nums.reshape((-1, 8))[:, 8-nbits:].reshape(-1)
    symbols = qam.modulate(bits)
        
    # Normalization to power = 1
    symbols /= np.sqrt(np.mean(np.real(symbols)**2 + np.imag(symbols)**2))
    return symbols

def generateZadoffChuSymbols(num_symbols, q=25):
    q = 25
    m = np.arange(num_symbols)
    seq = -np.pi * q * m * (m+1) / num_symbols
    symbols = np.cos(seq) + 1j*np.sin(seq)
    
    # Normalization to power = 1
    symbols /= np.sqrt(np.mean(np.real(symbols)**2 + np.imag(symbols)**2))
    return symbols

def receiveUSRP(num_samples, num_samples_per_buf, streamer, metadata):
    recv_buffer = np.zeros((1, num_samples_per_buf), dtype=np.complex64)
    recv_buffer_ = np.zeros((1, num_samples%num_samples_per_buf), dtype=np.complex64)
    
    num_buffers = num_samples // num_samples_per_buf
    samples = np.zeros(num_samples, dtype=np.complex64)
    for i in range(num_buffers):
        streamer.recv(recv_buffer, metadata)
        samples[i*num_samples_per_buf:(i+1)*num_samples_per_buf] = recv_buffer[0]
    streamer.recv(recv_buffer_, metadata)
    samples[num_buffers*num_samples_per_buf:] = recv_buffer_[0]
    
    return samples

def zeroMean(signal):
    signal -= np.mean(signal.signal)
    return signal

def interleave(symbols, sequence, decode=False):
    if decode:
        unshuf = np.zeros_like(sequence)
        unshuf[sequence] = np.arange(sequence.size)
        sequence = unshuf
    return symbols[sequence]

def plot(signal, plot=True):
    signal = signal.flatten()
    plt.plot(np.real(signal), label='Real')
    plt.plot(np.imag(signal), label='Imag')
    if plot:
        plt.legend()
        plt.show()

def scatter(signal, plot=True):
    signal = signal.flatten()
    plt.scatter(np.real(signal), np.imag(signal))
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature-Phase')
    if plot:
        plt.show()


class virtualChannel:
    def __init__(self, snrdB=0, channel=False, delay=False):
        self.snr = 10**(snrdB / 10)
        self.channel = channel
        self.delay = delay
    
    def __call__(self, signal):
        x = signal.signal
        
        if self.channel == True:
            h = np.random.normal(0, 0.5**0.5) + 1j * np.random.normal(0, 0.5**0.5)
        else:
            h = 1
        y = h * x
            
        sig_power = np.mean(np.abs(y)**2)
        no = (sig_power/(2*self.snr)) ** 0.5
        n = np.random.normal(0, no, x.shape) + 1j * np.random.normal(0, no, x.shape)
        y = y + n
        
        if self.delay == True:
            delay = np.random.randint(0, y.size)
            y = np.concatenate([y[delay:], y[:delay]])
            
        return USRP_signal(initial_symbols=y,
                                tone_Fs=signal.tone_Fs,
                                oversample_rate=signal.taps_per_symbol)
        
