import matplotlib.pyplot as plt
import time
import torch

import simulator.dsp.utils as utils
from simulator.fibres import SingleFibreParameters, NonLinearFibre
from simulator.datagen import SingleModeFibreGenerator, SingleModeParameterRange, SignalGenerator

sps = 4
alpha = 0.01
pulse_dur = 64
nSymbs = 128
sampling_freq = 12e10
freq_sep = 25e9
M = 4
nModes = 2
batch_size = 32768


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
sigs = utils.set_power(sigs, 3, dim=1, mode_dim=2)
sigs = sg.add_phase_noise(sigs)
t2 = time.time()
print(f"Done in {t2 - t1}")


print("Loading fibre")
t1 = time.time()
fibreB = NonLinearFibre(x, sg.nSamples, sg.freq_bins, batch_size=batch_size, device=torch.device("cuda"))
t2 = time.time()
print(f"Done in {t2 - t1}")


print("Simulating")
t1 = time.time()
AB = fibreB.simulate(sigs)
noise = torch.randn_like(AB)
noise = utils.set_power(noise, -30, dim=1, mode_dim=2)
# print(utils.get_power(AB, dim=1, mode_dim=2))
# print(utils.get_osnr(AB, noise, sampling_freq, dim=1, mode_dim=2))
AB += noise
t2 = time.time()
print(f"Done in {t2 - t1}")


print(F"Total: {time.time() - total_t1}s")