import numpy as np
import uhd
from threading import Thread

class USRPReceiver(Thread):
    def __init__(self, usrp, num_samples, carrier_frequency, sampling_rate, gain, channels, transmission_time):
        super().__init__()
        self.usrp = usrp
        self.num_samples = num_samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        
        for c in channels:
          usrp.set_rx_rate(self.sampling_rate, c)
          usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
          usrp.set_rx_gain(self.gain, c)
          usrp.set_rx_dc_offset(False, c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = channels
        self.streamer = self.usrp.get_rx_stream(st_args)
        self.num_samples_per_frame = self.streamer.get_max_num_samps()
        self.transmission_time = transmission_time
    
    
    def receiveUSRP(self, num_samples, metadata):
        num_channels = len(self.channels)

        recv_buffer = np.zeros((num_channels, self.num_samples_per_frame), dtype=np.complex64)
        recv_buffer_ = np.zeros((num_channels, num_samples%self.num_samples_per_frame), dtype=np.complex64)
        
        num_buffers = num_samples // self.num_samples_per_frame
        samples = np.zeros((num_channels, num_samples), dtype=np.complex64)
        for i in range(num_buffers):
            self.streamer.recv(recv_buffer, metadata)
            samples[:, i*self.num_samples_per_frame:(i+1)*self.num_samples_per_frame] = recv_buffer
        self.streamer.recv(recv_buffer_, metadata)
        samples[:, num_buffers*self.num_samples_per_frame:] = recv_buffer_
        
        return samples
    
    
    def run(self):
        # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(self.transmission_time)
        self.streamer.issue_stream_cmd(stream_cmd)
        
        metadata = uhd.types.RXMetadata()

        recv_buffer = np.zeros((len(self.channels), self.num_samples_per_frame), dtype=np.complex64)
        while self.streamer.recv(recv_buffer, metadata):
            pass

        NUM_INITIAL_IMPULSE_NOISE = 600
        self.rcv_samples = self.receiveUSRP(self.num_samples, metadata)[:, NUM_INITIAL_IMPULSE_NOISE:]
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stream_cmd)
        
        
class USRPTransmitter(Thread):
    def __init__(self, usrp, samples, carrier_frequency, sampling_rate, gain, channels, transmission_time):
        super().__init__()
        self.usrp = usrp
        assert samples.shape[0] == len(channels), "sample.shape[0] != # channel"
        self.samples = samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        
        for c in channels:
          self.usrp.set_tx_rate(self.sampling_rate, c)
          self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
          if self.gain is None:
              self.gain = 0
          self.usrp.set_tx_gain(self.gain, c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = channels
        self.streamer = self.usrp.get_tx_stream(st_args)
        self.transmission_time = transmission_time
    
    def run(self):
        TX_DELAY = 2000 / self.sampling_rate

        # Transmit Samples
        metadata = uhd.types.TXMetadata()
        metadata.has_time_spec = True
        metadata.start_of_burst = True 
        metadata.end_of_burst = True
        metadata.time_spec = uhd.types.TimeSpec(self.transmission_time + TX_DELAY)
        self.streamer.send(self.samples, metadata)
        
        # Help the garbage collection
        self.streamer = None
        
        
def sendAndReceive(usrp, signal, power, Fc, Fs, Tx_gain=0, Rx_gain=10, Tx_chan=[0, 1], Rx_chan=[0, 1]):
    signal *= np.sqrt(power)
    sending = signal.astype(np.complex64)

    INIT_DELAY = 0.2
    transmission_time = usrp.get_time_now().get_real_secs() + INIT_DELAY
    
    rx_thread = USRPReceiver(
        usrp=usrp,
        num_samples=int(signal.shape[-1] * 3),
        carrier_frequency=Fc,
        sampling_rate=Fs, 
        gain=Rx_gain,
        channels=Rx_chan,
        transmission_time=transmission_time
    )
    tx_thread = USRPTransmitter(
        usrp=usrp,
        samples=sending,
        carrier_frequency=Fc,
        sampling_rate=Fs, 
        gain=Tx_gain,
        channels=Tx_chan,
        transmission_time=transmission_time
    )
    rx_thread.start()
    tx_thread.start()
    tx_thread.join()
    rx_thread.join()
    
    rcv_signal = rx_thread.rcv_samples
    rcv_signal -= np.mean(rcv_signal, axis=-1, keepdims=True)
    rcv_signal /= np.sqrt(power)
    
    return rcv_signal
