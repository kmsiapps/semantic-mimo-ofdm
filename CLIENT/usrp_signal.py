import numpy as np
import math
import scipy
import uhd
from threading import Thread
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.style as mplstyle
from IPython import display
from modulate import modulate, modulations
import time

class USRP_signal():
    def __init__(self, initial_symbols, tone_Fs, oversample_rate=1):
        self.signal = initial_symbols
        self.tone_Fs = tone_Fs
        self.taps_per_symbol = oversample_rate
        
    def adjustSamplingRate(self, sampling_rate):
        '''
        Adjust sampling rate of self.frame
        '''
        if sampling_rate % self.taps_per_symbol and self.taps_per_symbol % sampling_rate:
            self.adjustSamplingRate(math.gcd(sampling_rate, self.taps_per_symbol))
        # Oversampling (pulsetrain)
        if sampling_rate > self.taps_per_symbol: 
            os_rate = sampling_rate // self.taps_per_symbol
            x = np.zeros(os_rate*len(self.signal), dtype = np.complex64)
            x[::os_rate] = self.signal
            self.signal = x
        # Undersampling (Decimation)
        else: 
            us_rate = self.taps_per_symbol // sampling_rate
            self.signal = self.signal[::us_rate]
        self.taps_per_symbol = sampling_rate
        
    def sampleToSymbols(self):
        self.adjustSamplingRate(1)
        return self.signal
        
    def __add__(self, x):
        if type(x) in (float, int, np.float_, np.complex_):
            self.signal += x
        else:
            assert self.taps_per_symbol == x.taps_per_symbol, "Addition Error: Sampling rate of both signal should be same"
            self.signal += x.signal
        return self
        
    def __sub__(self, x):
        if type(x) in (float, int, np.float_, np.complex_, np.complex64):
            self.signal -= x
        else:
            assert self.taps_per_symbol == x.taps_per_symbol, "Subtraction Error: Sampling rate of both signal should be same"
            self.signal -= x.signal
        return self
        
    def __mul__(self, x):
        assert type(x) in (float, int, np.float_, np.complex_), "Multiplication Error: Multiplication of (signal * constant) only is supported"
        self.signal * x
        return self
    
    def __truediv__(self, x):
        assert type(x) in (float, int, np.float_, np.complex_), "Division Error: Division of (signal / constant) only is supported"
        self.signal / x
        return self
    
    def __float__(self):
        return self.signal
    
    @property
    def size(self):
        return self.signal.size
    
    @property
    def shape(self):
        return self.signal.shape
    
    @property
    def dtype(self):
        return self.signal.dtype
    
    
    def plot(self, plot=True, grid=False):
        signal = self.signal.flatten()
        plt.plot(np.real(signal), label='Real')
        plt.plot(np.imag(signal), label='Imag')
        if grid:
            plt.xticks(np.arange(0, self.size+1, self.taps_per_symbol))
            plt.grid()
        if plot:
            plt.legend()
            plt.show()
    
    def scatter(self, plot=True):
        signal = self.signal.flatten()
        plt.scatter(np.real(signal), np.imag(signal))
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature-Phase')
        if plot:
            plt.show()
            
    def spectrum(self, plot=True):
        signal = self.signal.flatten()
        spectrum = np.fft.fftshift(np.fft.fft(signal))
        # spectrum = np.fft.fft(signal)
        spectrum_dB = 20 * np.log10(np.abs(spectrum))
        freq = np.fft.fftfreq(self.size, 1/(self.tone_Fs*self.taps_per_symbol))
        plt.plot(np.fft.fftshift(freq), spectrum_dB)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum (dB)')
        if plot:
            plt.show()
        