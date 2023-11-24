import torch
import os
import numpy as np
import math

from .dsp import utils as DSPUtils
from .fibres import NonLinearFibre, ParameterFields, BatchFibreParams, SingleFibreParameters


class SingleModeParameterRange():
    def __init__(self):
        self.attenuation = (0.13, 0.25) # dB/km
        self.dmd = (0, 1) # ps/sqrt(km)
        self.disp = (10, 25) # ps/nm/km
        self.slope = (0, 0.12) # ps/nm^2/km
        self.nl_coef = (0.5, 1.5) # /W/km
        self.length = (5e4, 5e4) # m
        self.xt_avg_km = (-40, 10) # dB/km
        self.dz = 1000 # m


class SingleModeFibreGenerator():
    def __init__(self, parameter_range: SingleModeParameterRange):
        self.pr = parameter_range
    
    def fibre_generator(self, device=torch.device("cpu"), batch_size=1):
        while True:
            params = BatchFibreParams()

            length = self.pr.length[0]
            dz = self.pr.dz
            xt_avg_km = self.uniform((batch_size, ), *self.pr.xt_avg_km, device=device)
            params[ParameterFields.FIBRE_LENGTH] = torch.full((batch_size,), length)
            params[ParameterFields.DZ] = torch.full((batch_size,), dz, device=device)
            params[ParameterFields.N_MODES] = torch.full((batch_size,), 2)

            params[ParameterFields.ATTENUATION] = (self.uniform((batch_size, 1), *self.pr.attenuation, device=device) / (10*np.log10(np.e)) / 1000).repeat(1,2)

            dmd = self.uniform((batch_size, ), *self.pr.dmd, device=device) * 1e-12 / np.sqrt(1000)
            corr_length  = 10**((10*np.log10((np.exp(2)-1)/(np.exp(2)+1))-(xt_avg_km))/10)*1000
            dgd_mean_total = dmd*np.sqrt(length)
            dgd_rms_total = dgd_mean_total/np.sqrt(8/3/np.pi)
            dgd_rms_unc_per_m = torch.sqrt(dgd_rms_total**2 / (2*corr_length**2*(torch.exp(-length/corr_length)+length/corr_length-1)))
            d_beta = np.sqrt(8/3/np.pi)*dgd_rms_unc_per_m
            params[ParameterFields.BETA1] = torch.stack((-d_beta/2, d_beta/2), dim=1)

            disp = (self.uniform((batch_size, 1), *self.pr.disp, device=device)).repeat(1,2) * 1e-6
            params[ParameterFields.BETA2] =  disp * (1550e-9**2) / (-2 * torch.pi * 3 * 10**8)
            slope = (self.uniform((batch_size, 1), *self.pr.slope, device=device)).repeat(1,2) * 1e3
            params[ParameterFields.BETA3] = 1550e-9**2/(2*torch.pi*3*10**8)**2 * (1550e-9**2*slope + 2*1550e-9*disp)

            # Coupling matrices
            steps = int(length / dz)
            rot_mat = torch.zeros((batch_size, steps, 2, 2), device=device, dtype=torch.complex64)

            var_to_xt = 10**((xt_avg_km + 10*np.log10(dz/1000))/10)
            rot_mat[:, :, 0, 0] = torch.randn((batch_size, steps), dtype=torch.float32)*1j*(1/np.sqrt(2)) # torch.cos(theta)
            rot_mat[:, :, 0, 1] = torch.randn((batch_size, steps), dtype=torch.complex64)*(1/np.sqrt(4)) # torch.sin(theta)
            rot_mat[:, :, 1, 0] = -torch.conj(rot_mat[:, :, 0, 1]) # torch.sin(theta)
            rot_mat[:, :, 1, 1] = torch.randn((batch_size, steps), dtype=torch.float32)*1j*(1/np.sqrt(2)) # -torch.cos(theta)
            c1 = torch.matrix_exp(rot_mat * torch.sqrt(var_to_xt)[:, None, None, None]) 

            # var_to_xt = 10**(xt_avg_km/10) / 40
            # theta = torch.rand((batch_size,steps), device=device) * 2 * torch.pi
            # rot_mat[:, :, 0, 0] = torch.cos(theta)
            # rot_mat[:, :, 0, 1] = torch.sin(theta)
            # rot_mat[:, :, 1, 0] = torch.sin(theta)
            # rot_mat[:, :, 1, 1] = -torch.cos(theta)
            # c2 = torch.matrix_exp(1j * rot_mat * torch.sqrt(var_to_xt)[:, None, None, None])

            params[ParameterFields.COUPLING_MATRICES] = c1


            # NL coefs
            nl_mat = torch.zeros(batch_size, 2, 2, device=device)
            nl_vals = self.uniform((batch_size,), *self.pr.nl_coef, device=device) / 1000
            # Main diagonal
            nl_mat[:,torch.arange(2), torch.arange(2)] = nl_vals[:, None]
            # Off diagonal
            nl_mat[:,-torch.arange(2)-1, torch.arange(2)] = 2 * nl_vals[:, None] / 3
            params[ParameterFields.NL_COEF] = nl_mat

            yield params


    def uniform(self, size, min_val, max_val, device=torch.device("cpu")):
        return torch.rand(size, device=device) * (max_val - min_val) + min_val


class LoadedFibreGenerator():
    def __init__(self, fibre_directory):
        self.fibre_files = os.listdir(fibre_directory)
        self.idx = 0


    def generate_batch(self, device=torch.device("cpu"), fibre_modifier=None, batch_size=1):
        """
        Generator function to provide fibre realisations.

        Parameters:
            fibre_directory: string
                The directory containing the fibre mat files
            fibre_modifier: func
                Function that takes in a set of fibre parameters and modifies them (e.g. dispersion)
            batch_size: int
                Size of the batch. Default: 1
            device: torch.device
                Default: torch.device("cpu")
        """

        while True:
            params = BatchFibreParams()

            # Loop through files to generate batch
            for _ in range(batch_size):
                fibre_path = os.path.join(fibre_directory, self.fibre_files[self.idx])
                fp = SingleFibreParameters(fibre_path)

                for key in params.keys():
                    params[key].append(fp[key])

                self.idx = (self.idx+1) % len(fibre_files)
            
            # Stack parameters into batches
            for key in params.keys():
                params[key] = torch.stack(params[key]).to(device=device)


            # Make sure all fibres are same dimensions (for batch processing purposes)
            assert torch.all(params[ParameterFields.DZ] == params[ParameterFields.DZ])
            assert torch.all(params[ParameterFields.FIBRE_LENGTH] == params[ParameterFields.FIBRE_LENGTH])
            assert torch.all(params[ParameterFields.N_MODES] == params[ParameterFields.N_MODES])
            
            yield params


class SignalGenerator():
    """
    Generates random Root Raised Cosine filtered QAM signals
    """

    def __init__(self, sps, alpha, pulse_dur, nSymbs, nModes, nChnls, freq_sep, sampling_freq, M, device=torch.device("cpu")):
        """
        Initialises filtering window and frequency and time bins for the signal.

        Parameters:
            sps: int
                Samples per symbol
            alpha: float
                Roll-off for Root Raised Cosine
            pulse_dur: int
                Number of taps in RRC filter 
            nSymbs: int
                Number of symbols to send in each signal
            nModes: int
                Number of modes used in each signal
            nChnls: int
                Number of WDM channels used in each signal
            freq_sep: int
                Separation in Hz between WDM channels
            sampling_freq: int
                Sampling frequency of the signal   
            M: int
                The length of one side of the QAM constellation (i.e. M=4 => 16QAM)
            device: torch.device
                Device to process data on. Default: torch.device("cpu")
        """
        self.sps = sps
        self.nSymbs = nSymbs
        self.nModes = nModes
        self.nChnls = nChnls
        self.freq_sep = freq_sep
        self.sampling_freq = sampling_freq
        self.M = M
        self.nSamples = sps * nSymbs
        self.device = device

        self.b = DSPUtils.create_rrc_filter(sps, alpha, pulse_dur, device)
        self.freq_bins, self.time_bins = DSPUtils.generate_bins(sampling_freq, self.nSamples, device)


    def signal_generator(self, fibre_generator, signals_per_fibre, num_fibres, pre_process=lambda x: x, post_process=lambda x: x, pilot_spacing=None):
        """
        Initialises a generator for producing batches of signals given a fibre generator

        Parameters:
            fibre_generator: Generator
                Generator that produces a BatchFibreParams class at each iteration
            signals_per_fibre: int 
                Number of signals to generate per fibre (to minimize fibre generation time if long)
            num_fibres: int
                Number of fibres in each batch produced by the fibre generator
            pre_process: func
                Prepares the signals for propagation through the fibre. Can include adding phase noise, setting power, etc. Takes in
                a batch of signals and outputs the pre-processed batch of signals
            post_process: func
                Processes the signal after propagation, which may include AWGN, normalisation, etc. Takes a batch of signals and outputs
                post-processed signals.
            pilot_spacing: int
                Specifies number of symbols between pilots. If None (default), 
        """
        while True:
            fibre_params = next(fibre_generator)    

            fibres = NonLinearFibre(fibre_params, self.nSamples, self.freq_bins, num_fibres, device=self.device)
            for _ in range(signals_per_fibre):
                launch_power = 0 + torch.rand((num_fibres,), device=self.device)*10
                common_offset = 0#((torch.rand((num_fibres, ), device=self.device)*2 - 1) * torch.pi)[:, None]

                tx_train = self.generate_cazac_seq().repeat((num_fibres, 1, 1))
                # tx_train = DSPUtils.set_power(tx_train, launch_power, dim=1, mode_dim=2)
                tx_train = self.add_phase_noise(DSPUtils.set_power(tx_train, launch_power, dim=1, mode_dim=2), common_offset)
                rx_train = fibres.simulate(tx_train)
                rx_train = post_process(rx_train)

                symbs = self.generate_symbs(pilots_spacing=pilot_spacing, batch_size=num_fibres)
                tx_data = self.generate_signal(symbs)
                offset = torch.randint(0, 20, (tx_data.shape[0],), device=self.device) * self.nSymbs * self.sps
                tx_data = self.add_phase_noise(DSPUtils.set_power(tx_data, launch_power, dim=1, mode_dim=2), common_offset, offset=offset)
                # tx_data = DSPUtils.set_power(tx_data, launch_power, dim=1, mode_dim=2)
                # tx_data = pre_process(tx_data)
                rx_data = fibres.simulate(tx_data)
                rx_data = post_process(rx_data)
                # yield tx_data, tx_train, symbs
                yield rx_data, rx_train, symbs


    def add_phase_noise(self, signal, common_offset, linewidth_min=1e4, linewidth_max=1e5, offset=0):
        """
        Adds laser phase noise to input signal. Adds a common phase offset to the entire signal, followed by phase
        noise generated via a random walk.

        Parameters:
            signal: torch.tensor, shape=(batch_size, nSamples, nModes), dtype=torch.complex64
            common_offset: torch.tensor, shape=(batch_size), dtype=torch.complex64
                Phase offset applied to entire signal
            linewidth_min: int
                The minimum linewidth for generating the phase noise. Default at 10kHz
            linewidth_max: int
                Maximum linewidth for generating phase noise. Default at 100kHz
            offset: int | torch.tensor, shape=(batch_size,), dtype=int
                Offset (in samples) between the training sequence and the data sequence
        
        Returns:
            signal: torch.tensor, shape=(batch_size, nSamples, nModes), dtype=torch.complex64
                Signal that has been exposed to laser phase noise
        """
        linewidth = linewidth_min + torch.rand((signal.shape[0], ), device=self.device) * (linewidth_max - linewidth_min)
        pn = self.get_phase_noise(linewidth, self.nSamples, common_offset, signal.shape[0], offset=offset)
        return signal * pn[:, :, None]
    

    def get_phase_noise(self, linewidth, num_samples, common_offset, batch_size=1, offset=0):
        """
        Gets the phase change in the form of a complex signal with amplitude 1 and varying argument.

        Parameters
            linewidth: torch.tensor, shape=(batch_size), dtype=torch.float32
                The linewidth of the laser, used to compute variance of the random walk
            num_samples: int
                Number of samples in frame. This is the length of the random walk
            common_offset: int |
        """
        ts = 1/self.sampling_freq
        var = 2 * torch.pi * linewidth * ts
        phase_offset = torch.randn((batch_size, ), device=self.device) * torch.sqrt(var * offset)
        phase_steps = torch.randn((batch_size, num_samples), device=self.device) * torch.sqrt(var)[:, None] 
        phase = torch.cumsum(phase_steps, dim=1) + phase_offset[:, None] + common_offset
        return torch.cos(phase) + 1j * torch.sin(phase)


    def add_noise(self, signal, noise_power_min, noise_power_max):
        noise_power = noise_power_min + torch.rand((signal.shape[0],), device=self.device) * (noise_power_max - noise_power_min)
        noise = torch.randn(signal.shape, dtype=torch.complex64, device=self.device)
        noise = DSPUtils.set_power(noise, noise_power, dim=1, mode_dim=2)
        return signal + noise


    def adc_clip(self, signal, clip_ratio=0.05, dim=0):
        """
        Performs clipping at the ADC

        Parameters:
            signal: torch.tensor, shape=(..., nSamples, ...), dtype=torch.complex64
                Signal to be clipped. Size at 'dim' is nSamples
            clip_ratio: float
                Ratio of signal to clip. Default: 0.05
            dim: int
                The dimension across which the signal varies over time. Default: 0
        """
        max_amp = torch.max(torch.abs(signal))
        min_amp = torch.min(torch.abs(signal))
        amp_diff = max_amp - min_amp
        clipped_max = max_amp - clip_ratio*amp_diff
        clipped_min = min_amp + clip_ratio*amp_diff
        print(clipped_max)
        print(clipped_min)
        i_max = torch.abs(signal) > clipped_max
        signal[i_max] = signal[i_max] / torch.abs(signal[i_max]) * clipped_max

        i_min = torch.abs(signal) < clipped_min
        signal[i_min] = signal[i_min] / torch.abs(signal[i_min]) * clipped_min

        return signal
    

    def ofdm_clip(self, signal, clip_ratio_db, dim=0):
        """
        Performs clipping at the transmitter, generally used in OFDM to lower peak-to-average-power-ratio.
        """
        clip_ratio = 10**(clip_ratio_db/20)
        power = torch.mean(torch.abs(signal)**2, dim=0, keepdim=True)
        A = clip_ratio * torch.sqrt(power)
        p = torch.abs(signal) > A
        signal[p] = (A * (signal/torch.abs(signal)))[p]
        return signal



    def generate_symbs(self, pilots_spacing=None, batch_size=1):
        xI = torch.randint(0, self.M, (batch_size,self.nSymbs,self.nModes,self.nChnls), device=self.device)
        xI = 2*xI - (self.M-1)
        xQ = torch.randint(0, self.M, (batch_size,self.nSymbs,self.nModes,self.nChnls), device=self.device)
        xQ = 2*xQ - (self.M-1)
        symbs = xI + 1j*xQ

        if self.M > 2:
            symbs /= (self.M - 1)
        
        # Insert pilots
        if pilots_spacing:
            # pol1 = [0.7071-0.7071j, 0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j, -0.7071+0.7071j, -0.7071+0.7071j, -0.7071+0.7071j, -0.7071-0.7071j, -0.7071+0.7071j, -0.7071+0.7071j, -0.7071-0.7071j,  0.7071-0.7071j, -0.7071+0.7071j, -0.7071-0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, -0.7071-0.7071j, -0.7071-0.7071j, -0.7071+0.7071j, -0.7071+0.7071j, -0.7071-0.7071j,  0.7071+0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, -0.7071-0.7071j, 0.7071+0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, 0.7071+0.7071j, 0.7071+0.7071j, 0.7071+0.7071j, 0.7071+0.7071j]
            # pol2 = [0.7071-0.7071j, -0.7071-0.7071j, -0.7071-0.7071j, 0.7071+0.7071j, 0.7071+0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j, 0.7071+0.7071j,  0.7071-0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, -0.7071-0.7071j, -0.7071-0.7071j,  0.7071-0.7071j, 0.7071-0.7071j, 0.7071+0.7071j, 0.7071-0.7071j, 0.7071-0.7071j, -0.7071+0.7071j, 0.7071+0.7071j, 0.7071-0.7071j, 0.7071-0.7071j, 0.7071+0.7071j]
            # pol1 = torch.tensor(pol1, dtype=torch.complex64, device=self.device)
            # pol2 = torch.tensor(pol2, dtype=torch.complex64, device=self.device)
            # pilot = torch.stack((pol1, pol2), dim=1)[None, :, :, None]
            # symbs[:, ::pilots_spacing] = pilot[:,:self.nSymbs//pilots_spacing].repeat((batch_size, 1, 1, 1))
            
            gen = torch.Generator(device=self.device)
            gen = gen.manual_seed(0)
            pI = torch.randint(0, 2, (self.nSymbs // pilots_spacing,self.nModes,self.nChnls), device=self.device, generator=gen).to(dtype=torch.float) * 2 - 1
            pI *= np.sqrt(2)/2
            pQ = torch.randint(0, 2, (self.nSymbs // pilots_spacing,self.nModes,self.nChnls), device=self.device, generator=gen).to(dtype=torch.float) * 2 - 1
            pQ *= np.sqrt(2)/2
            pilot = pI + 1j*pQ
            symbs[:, ::pilots_spacing] = pilot.repeat((batch_size, 1, 1, 1))   
                
        return symbs


    def generate_cazac_seq(self):
        M = 61
        theta = M * torch.pi * torch.arange(self.nSymbs, device=self.device)**2 / self.nSymbs
        cazac_seq = torch.cos(theta) + 1j * torch.sin(theta)

        training_seq = torch.zeros(self.nSymbs, self.nModes, dtype=torch.complex64, device=self.device)
        for i in range(0,self.nModes//2):
            s = math.ceil(self.nSymbs/self.nModes)
            training_seq[:,2*(i-1)] = torch.roll(cazac_seq, s*2*(i-1))
            training_seq[:,2*i-1] = torch.roll(cazac_seq, s*(2*i-1))
        
        return DSPUtils.resample_poly(training_seq, self.sps, 1, self.b, dim=0)
    

    def generate_bpsk_seq(self):
        gen = torch.Generator(device=self.device)
        gen = gen.manual_seed(0)
        c = torch.randint(0, 2, (self.nSymbs, ), device=self.device, generator=gen).to(dtype=torch.float) * 2 - 1
        training_seq = torch.zeros(self.nSymbs, self.nModes, dtype=torch.complex64, device=self.device)
        for i in range(0,self.nModes):
            training_seq[i::self.nModes, i] = c[i::self.nModes]
        training_seq = 1/np.sqrt(self.nSymbs) * torch.fft.ifft(training_seq, dim=0)
        return DSPUtils.resample_poly(training_seq, self.sps, 1, self.b, dim=0)


    def generate_signal(self, symbs):
        """
        Generates a random QAM signal to be sent in the time domain.
        
        Returns: 
            sig: torch.tensor, shape (nSamples, nModes), dtype complex128
                The complex signal to be sent
            symbs: torch.tensor, shape (nSymbs, nModes, nChnls), dtype complex128
                The actual set of symbols that was sent
        """    
        freq_bins = self.freq_bins[:, None]
        nearest_freq_idx = torch.abs(freq_bins - ((-(self.nChnls//2) + torch.arange(self.nChnls, device=self.device)) * self.freq_sep)).argmin(dim=0)
        near_freq = self.freq_bins[nearest_freq_idx]
        symbs_index = torch.ones((self.nSymbs,), dtype=torch.bool)
        symbs_index[::32] = 0
        symbs[:, symbs_index, :, :] = DSPUtils.normalize(symbs[:, symbs_index, :, :], dim=1)
        shaped_signal = DSPUtils.resample_poly(symbs, self.sps, 1, self.b, dim=1)

        theta = 2*torch.pi*near_freq*self.time_bins[:,None]
        w = torch.cos(theta) + 1j*torch.sin(theta)
        sig = torch.sum(w[None, :, None, :] * shaped_signal, axis=-1)
        #sig = torch.sum(torch.exp(1j*2*torch.pi*near_freq*self.time_bins[:, None])[None, :, None, :] * shaped_signal, axis=-1)
        return sig


    def get_Rx_signal(self, sig):
        """
        Determines the received signal

        Parameters:
            sig: torch.tensor, shape (nSamples, nModes), dtype complex128
                The complex signal that was received in the time domain

        Returns:
            symbs: torch.tensor, shape (nSymbs, nModes, nChnls), dtype complex128
                The actual set of symbols that was received
        """
        symbs = torch.zeros((self.nSymbs,self.nModes,self.nChnls),dtype=torch.complex64)
        sig = sig.unsqueeze(0)
        for channel in range(self.nChnls):
            nearest_freq_idx = torch.abs(self.freq_bins - ((-(self.nChnls//2) + channel)*self.freq_sep)).argmin()
            near_freq = self.freq_bins[nearest_freq_idx]

            for k in range(self.nModes):
                temp = DSPUtils.resample_poly(torch.exp(1j*2*np.pi*(-near_freq)*self.time_bins)*sig[:,:,k], 1, self.sps, self.b, dim=1)
                symbs[:,k,channel] = temp[0]
        
        return symbs


if __name__ == "__main__":
    sps = 4
    alpha = 0.01
    pulse_dur = 401
    nSymbs = 1024
    sampling_freq = 12e10
    freq_sep = 25e9
    M = 4
    nModes = 2

    batch_size = 64

    import matplotlib.pyplot as plt
    # sg = SignalGenerator(sps, alpha, pulse_dur, nSymbs, nModes, 1, freq_sep, sampling_freq, M, torch.device("cuda"))
    # symbs = sg.generate_symbs(batch_size)
    # sigs = sg.generate_signal(symbs)
    # pn = sg.get_phase_noise(torch.tensor([2e5]), nSymbs*sps, batch_size)[:,:,None]
    # print(pn.shape)
    # print(sigs.shape)
    # sigs *= pn
    # symbs_rx = sg.get_Rx_signal(sigs[5])[200:-200].flatten()
    # plt.scatter(symbs_rx.real, symbs_rx.imag)
    # plt.show()
    # quit()
    
    # for i in range(5):
    #     plt.plot(range(nSymbs*sps),pn[i])
    # plt.show()
    # quit()

    import time
    total_t1 = time.time()

    print("Generating Fibre")
    t1 = time.time()
    fibre_gen = SingleModeFibreGenerator(SingleModeParameterRange())
    fibre_generator = fibre_gen.fibre_generator(device=torch.device("cuda"), batch_size=batch_size)
    x = next(fibre_generator)
    t2 = time.time()
    print(f"Done in {t2 - t1}")

    print("Generating transmit signals")
    t1 = time.time()
    sg = SignalGenerator(sps, alpha, pulse_dur, nSymbs, nModes, 1, freq_sep, sampling_freq, M, torch.device("cuda"))
    symbs = sg.generate_symbs(pilots_spacing=16,batch_size=batch_size)
    sigs = sg.generate_signal(symbs)
    sigs = DSPUtils.set_power(sigs, 3, dim=1, mode_dim=2)
    sigs = sg.add_phase_noise(sigs)
    # pn = sg.get_phase_noise(torch.tensor([2e4], device=torch.device("cuda")), nSymbs*sps, batch_size)[:,:,None]
    # sigs *= pn
    t2 = time.time()
    print(f"Done in {t2 - t1}")

    
    fibre_directory = "FibreGeneration\\Fibres\\nModes=1_7_150" 

    fibre_files = os.listdir(fibre_directory)

    # print("Single sims")
    # Single sims
    # rx_sigsS = []
    # time_taken = 0
    # sigs = sigs.to(device=torch.device("cpu"))
    # for i,sig in enumerate(sigs):
        
    #     fibre_path = os.path.join(fibre_directory, fibre_files[i % len(fibre_files)])
    #     fp = FibreParameters(fibre_path)
    #     fibreS = NL1(fp, sg.nSamples, sg.freq_bins)

    #     t1 = time.time()
    #     rx_sigsS.append(fibreS.simulate(sig, 0, 100))
    #     t2 = time.time()
    #     time_taken += t2 - t1
    # print(f"Done in {time_taken}")
    #AS = torch.stack(rx_sigsS)  

    # Batched sims
    print("Loading fibre")
    # fibre_generator = LoadedFibreGenerator(fibre_directory)
    # fibre_gen = fibre_generator.generate_batch(device=torch.device("cuda"), batch_size=batch_size)
    # x = next(fibre_gen)
    # for k in x:
    #     if k != ParameterFields.COUPLING_MATRICES:
    #         print(f"{k}: {x[k][0]}")
    t1 = time.time()
    fibreB = NonLinearFibre(x, sg.nSamples, sg.freq_bins, batch_size=batch_size, device=torch.device("cuda"))
    t2 = time.time()
    print(f"Done in {t2 - t1}")
    # sigs = sigs.to(device=torch.device("cuda"))


    print("Simulating")
    t1 = time.time()
    AB = fibreB.simulate(sigs)
    noise = torch.randn_like(AB)
    noise = DSPUtils.set_power(noise, -30, dim=1, mode_dim=2)
    print(DSPUtils.get_power(AB, dim=1, mode_dim=2))
    print(DSPUtils.get_osnr(AB, noise, sampling_freq, dim=1, mode_dim=2))
    AB += noise
    t2 = time.time()

    # Test if simulating batch is same as single simulation

    # single_params = x[10]
    # for k in single_params.params:
    #     single_params[k] = single_params[k].unsqueeze(0)
    # fibreS = NonLinearFibre(single_params, sg.nSamples, sg.freq_bins, batch_size=1, device=torch.device("cuda"))
    # AS = fibreS.simulate(sigs[10].unsqueeze(0), 0, 100)
    # print(AB[10] - AS)

    # pred_symbs = sg.get_Rx_signal(sigs[10].to(device=torch.device("cuda")))[100:-100,:].flatten()
    # plt.scatter(pred_symbs.real, pred_symbs.imag)
    # plt.show()
    # quit()

    print(f"Done in {t2 - t1}")

    print(F"Total: {time.time() - total_t1}s")

    from dsp.algorithms import OptimalLinear, OptimalDBP, DBPAndLinear
    single_params = x[10]
    for k in single_params.params:
        single_params[k] = single_params[k].unsqueeze(0)#.to(device=torch.device("cpu"))
    # single_params[ParameterFields.ATTENUATION][:] = 0
    # single_params[ParameterFields.BETA1][:] = 0
    # single_params[ParameterFields.BETA2][:] = 0
    # single_params[ParameterFields.BETA3][:] = 0
    #single_params[ParameterFields.NL_COEF][:] = 0
    #single_params[ParameterFields.NL_COEF][:] = torch.tensor([[0.0012, 0.0008], [0.0012, 0.0008]], device=torch.device("cuda"))
    # for i in range(100):
    #     single_params[ParameterFields.COUPLING_MATRICES][:,i] = torch.eye(2)

    # fibreS = NonLinearFibre(single_params, sg.nSamples, sg.freq_bins, batch_size=1, device=torch.device("cuda"))
    # s = DSPUtils.set_power(sigs[10], 0)
    # AS = fibreS.simulate(s.unsqueeze(0), 0, 100)
    AS = AB[10].unsqueeze(0)

    #filter = DBPLS(single_params, nSymbs * sps, sg.freq_bins.to(device=torch.device("cpu")), 1, 1000, 0)
    #pred = filter.simulate(AS)
    # pred_symbs = sg.get_Rx_signal(AS.squeeze(0))[100:-100,:].flatten()
    # plt.scatter(pred_symbs.real, pred_symbs.imag)
    # plt.show()
    # print(pred - sigs[10].to(device=torch.device("cpu")))

    # single_sig = AB[10].to(device=torch.device("cpu"))
    single_sig = AS.squeeze(0).to(device=torch.device("cpu"))
    for k in single_params.params:
        single_params[k] = single_params[k].squeeze(0).to(device=torch.device("cpu"))
    filter = OptimalDBP(single_params, nSymbs * sps, sg.freq_bins.to(device=torch.device("cpu")))
    pred = filter.simulate(single_sig)
    pred_symbs = sg.get_Rx_signal(pred.to(device=torch.device("cuda")))[100:-100].flatten()
    plt.scatter(pred_symbs.real, pred_symbs.imag)
    # act_symbs = sg.get_Rx_signal(single_sig.to(device=torch.device("cuda")))[100:-100].flatten()
    # plt.scatter(act_symbs.real, act_symbs.imag)
    plt.show()
    #print(pred - sigs[10].to(device=torch.device("cpu")))
