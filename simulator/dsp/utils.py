import torch
import torch.nn.functional as F
import numpy as np
import sys
import math


# Adapted from https://gist.github.com/thomasbrandon/63d609f37f8e73c56f5a4c76260aeb28
def ufd_out_len(n_h, n_in, up, down):
    nt = n_in + (n_h + (-n_h % up)) // up - 1
    nt *= up
    need = nt // down
    if nt % down > 0: need += 1
    return need

# Preprocessing from scipy's resample_poly
def resample_poly_pre(n_x, up, down, filt):
    """
    Parameters:
        x: torch.tensor, shape (batch_size, signal_len)
    """
    # In our case one of up or down is 1 so this block does nothing
    #########################
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_
    #########################

    # Only one of these will be relevant (same reason as before)
    ##################################
    # Number of samples if upsampling
    n_out = n_x * up
    # Number of symbols if downsampling
    n_out = n_out // down + bool(n_out % down)
    ##################################
    
    if filt.ndimension() != 1: raise ValueError("Filter should be a 1d tensor")
    n_filt = len(filt)
    # Filter should have odd length
    half_len = (n_filt-1) // 2
    # Scale filter by upsampling amount
    h = filt * up
    
    # Zero-pad filter to put output samples at center
    n_pre_pad = (down - half_len % down)
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while ufd_out_len(n_filt + n_pre_pad + n_post_pad, n_x, up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = F.pad(h, (n_pre_pad, n_post_pad))
    n_pre_remove_end = n_pre_remove + n_out
    # Slice of data that is relevant (taken from the middle "n_out" slice)
    remove = slice(n_pre_remove,n_pre_remove_end)
    return h,up,down,remove


# And a resample_poly that takes an upfirdn to use.
def resample_poly(x, up, down, filt, dim=0):
    n_x = x.shape[dim]
    h,up,down,remove = resample_poly_pre(n_x, up, down, filt)
    y, perm, inv = upfirdn_dot(h, x, up, down, dim)
    # y, perm, inv = upfirdn_conv(h, x, up, down, dim)
    # y, perm, inv = upfirdn_zs(h, x, up, down, dim)
    # print(upfirdn_dot(h, x, up, down, dim)[0] - upfirdn_zs(h, x, up, down, dim)[0])
    y = y[...,remove]

    out_shape = list(x.shape)
    out_shape[dim] = int(out_shape[dim] * up / down)
    out = y.view([out_shape[p] for p in perm])
    out = torch.permute(out, inv)
    return out

# Zero-stuffing version
def upfirdn_zs(h, x, up, down, dim): 
    y, perm, inv = upfir_zs(h, x, up, dim)
    # If downsampling, take outputs at every interval
    y_out_slice = [slice(None)] * y.ndim
    y_out_slice[dim] = slice(None, None, down)
    return y[y_out_slice], perm, inv

def upfir_zs(h, x, up, dim): # Zero-padding version
    # Output shape
    out_shape = list(x.shape)
    out_shape[dim] *= up

    # Zero stuff x by spacing each value out by "up" zeros
    x_zp = torch.zeros(*out_shape, dtype=x.dtype, device=x.device)
    s = [slice(None)] * x_zp.ndim
    s[dim] = slice(None, None, up)
    x_zp[s] = x
    
    # Reshape into batches for conv
    new_shape = (-1, 1, x_zp.shape[dim])
    perm = (*range(dim), *range(dim + 1, x_zp.ndim), dim)
    x_zp_c = x_zp.permute(perm).reshape(new_shape)

    # Compute inverse of permutation for acquiring output
    perm = torch.tensor(perm)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)

    # Reverse coefficients and add out/in channel dimensions
    n_h = h.shape[0]
    h_c = h.flip(0)[None,None,...]
    pad = n_h-1

    # Apply convolution
    n_x = x.shape[dim]
    n_out = ufd_out_len(n_h, n_x, up, 1)

    y = F.conv1d(x_zp_c, h_c, padding=pad)[:,0,:n_out]

    return y, perm, tuple(inv)


def upfir_conv(h, x, up, dim):
    n_h,n_x = h.shape[0],x.shape[dim]
    h_pad = F.pad(h, (0, -n_h % up)) # Pad to multiple of up
    n_h_pp = len(h_pad) // up # number of taps per phase
    h_pp = h_pad.view(1, 1, n_h_pp, -1).transpose(0,1).flip(2) # Filters per phase
    n_y_pp = n_x + n_h_pp-1 # Outputs per phase
    # Up to here has been modified for batches
    # Can easily compare with upfirdn_zs as I have already verified that it is close to original resample_poly=

    # Reshape into batches for conv
    new_shape = (-1, 1, x.shape[dim])
    perm = (*range(dim), *range(dim + 1, x.ndim), dim)
    x_zp_c = x.permute(perm).reshape(new_shape)

    # Compute inverse of permutation for acquiring output
    perm = torch.tensor(perm)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)


    y_pp = torch.zeros((x_zp_c.shape[0], n_y_pp, up), dtype=torch.complex64, device=x.device)
    # This can be vmapped over i if needed
    for i in range(up): # Apply phases
        y_pp[:,:,i] = F.conv1d(x_zp_c, h_pp[...,i], padding=n_h_pp-1)[:,0,:]

    y = y_pp.view(x_zp_c.shape[0],-1) # Interleave outputs
    return y, perm, tuple(inv)


def upfirdn_conv(h, x, up, down, dim):
    y, perm, inv = upfir_conv(h, x, up, dim)
    # If downsampling, take outputs at every interval
    y_out_slice = [slice(None)] * y.ndim
    y_out_slice[dim] = slice(None, None, down)
    return y[y_out_slice], perm, inv

    # y = upfir_conv(h, x, up)
    # return y[...,::down]


def upfirdn_dot(h, x, up, down, dim):
    n_h,n_x = len(h), x.shape[dim]
    # Split filter into up phases, padding so even lengths, coefficients in each phase should be reversed
    hpad = F.pad(h, (0, -n_h%up))
    n_hpp = len(hpad)//up
    hpp = hpad.view(n_hpp, -1).t().flip(1)

    # Reshape into batches for conv
    new_shape = (-1, x.shape[dim])
    perm = (*range(dim), *range(dim + 1, x.ndim), dim)
    x_zp_c = x.permute(perm).reshape(new_shape)

    # Compute inverse of permutation for acquiring output
    perm = torch.tensor(perm)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)

    # Create padded input
    xpad = F.pad(x_zp_c, (n_hpp-1, n_hpp-1, 0, 0))
    # Create output array and fill it
    n_y = ufd_out_len(n_h, n_x, up, down)
    y = torch.zeros((x_zp_c.shape[0], n_y), dtype=x.dtype, device=x.device)
    for i_y in range(n_y):
        i_x = i_y * down // up # Input idx at which to filter
        i_hpp = (i_y * down) % up # Filter phase to use - i_y*down gives non-downsampled index
        # Apply FIR to sample by calculating dot product
        y[:,i_y] = xpad[:,i_x:i_x+n_hpp].matmul(hpp[i_hpp,:])
        #y[:, i_y] = hpp[i_hpp,:].dot(xpad[i_x:i_x+n_hpp])
    return y, perm, tuple(inv)


def resample_fourier(signal, num, dim=0):
    """
    Resamples the signal using the Fourier method (same as scipy implementation)

    Parameters:
        Signal: torch.tensor, shape (a, ..., dim, ...)
            Signal to be resampled
        num: int
            The new number of samples to resample to
        dim: int
            The dimension over which to resample. Default: 0
    
    Returns:
        Resampled signal
    """
    Nx = signal.shape[dim]
    X = torch.fft.fft(signal, dim=dim)

    new_shape = list(signal.shape)
    new_shape[dim] = num
    Y = torch.zeros(new_shape, dtype=X.dtype, device=X.device)

    N = min(num, Nx)
    nyq = N // 2 + 1
    sl = [slice(None)] * signal.ndim
    
    # Copy positive frequency components
    sl[dim] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    # Copy negative frequency components
    if N > 2:
        sl[dim] = slice(nyq - N, None)
        Y[tuple(sl)] = X[tuple(sl)]

    if N % 2 == 0:
        # If downsampling
        if num < Nx: 
            # select the component of Y at frequency +N/2,
            # add the component of X at -N/2
            sl[dim] = slice(-N//2, -N//2 + 1)
            Y[tuple(sl)] += X[tuple(sl)]

        # If upsampling
        elif Nx < num:
            # select the component at frequency +N/2 and halve it
            sl[dim] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            temp = Y[tuple(sl)]
            # set the component at -N/2 equal to the component at +N/2
            sl[dim] = slice(num-N//2, num-N//2 + 1)
            Y[tuple(sl)] = temp
    
    # Back to time domain and adjust magnitude
    y = torch.fft.ifft(Y, dim=dim)
    y *= (float(num)/float(Nx))
    return y


def apply_filter(signal, window, dim=0):
    """
    Applies FIR filter over a signal. Implements using the fourier method.

    Parameters:
        signal: torch.tensor
        window: torch.tensor
        dim: int
            Default = 0
    
    Returns: 
        Filtered signal
    """
    X = torch.fft.fft(signal, dim=dim)

    Nx = signal.shape[dim]
    Nw = window.shape[0]
    Ndiff = Nx - Nw
    window = F.pad(window, (Ndiff//2, Ndiff//2), "constant", 0)
    W = torch.fft.fft(window)# * 4
    W_shape = [1 for _ in range(X.ndim)]
    W_shape[dim] = W.shape[0]

    Y = X * W.view(*W_shape)
    y = torch.fft.ifft(Y, dim=dim)

    return y


def normalize(signal, dim=0):
    """
    Normalizes the power of the input signal over the dimension specified.

    Parameters:
        signal: torch.tensor
            Signal to normalize
        dim: int
            Dimension over which to take average power - this will most likely be the time or frequency dimension.
            Default: 0
    """
    power = signal.abs() ** 2
    power = power.mean(dim=dim, keepdim=True).sqrt()
    signal /= power
    return signal


# Get and set power aren't working (:
def set_power(signal, power, dim=0, mode_dim=1):
    """
    Sets the power of a signal

    Parameters:
        signal: torch.tensor
        power: float
            The power in dBms to set the signal to
        dim: int
            The dimensions over which to set power
        mode_dim: int
            The dimension of the separate modes
    
    Returns:
        signal: torch.tensor
            The signal set to the given power
    """
    # Divide power amongst modes
    power -= 10 * np.log10(signal.shape[mode_dim])
    signal_power = 1000*torch.mean((torch.abs(signal))**2, dim=dim, keepdim=True)
    
    coef = torch.sqrt(10**(power/10) / signal_power)
    signal *= coef
    return signal


def get_power(signal, dim=0, mode_dim=1):
    """
    Gets the power of a given signal.

    Parameters:
        Signal: torch.tensor
        dim: int
            The dimension over which the signal varies (either time or frequency)
        mode_dim: int
            The dimension of the separate modes
    
    Returns:
        power: float
            The power of the signal
    """
    return 10*torch.log10(torch.sum(torch.mean((torch.abs(signal))**2, dim=dim, keepdim=True), dim=mode_dim, keepdim=True)/1e-3)


def get_osnr(signal, noise_signal, sample_freq, dim=0, mode_dim=1):
    """
    Gets the OSNR given the noise signal
    """
    signal_power = get_power(signal, dim=dim, mode_dim=mode_dim)
    noise_power = 10*torch.log10(torch.sum(torch.mean(torch.abs(noise_signal)**2, dim=dim, keepdim=True), dim=mode_dim, keepdim=True)*1000/sample_freq*12.5e9)
    return signal_power - noise_power


def create_rrc_filter(sps, alpha, pulse_dur, device):
    """     
    Generates the root raised cosine filter.

    Parameters:
        sps: int
            Samples per symbol
        alpha: float
            Roll-off for Root Raised Cosine
        pulse_dur: int
            Duration of each pulse
    
    Returns: np.ndarray, shape (pulse_dur * sps * 2,)
        The RRC window
    """
    n = torch.arange(-pulse_dur*sps,pulse_dur*sps+1)

    # Isn't this just 1/4?
    eps = torch.abs(n[0]-n[1])/4
    
    with np.errstate(divide='ignore', invalid='ignore'):
        b = 1/sps*((torch.sin(torch.pi*n/sps*(1-alpha)) +  4*alpha*n/sps*torch.cos(torch.pi*n/sps*(1+alpha)))/(torch.pi*n/sps*(1-(4*alpha*n/sps)**2)))
        b = b.to(dtype=torch.complex64, device=device)
        idx1 = torch.abs(n) < eps
        b[idx1] = 1/sps*(1+alpha*(4/np.pi-1))
        idx2 = torch.abs(torch.abs(n)-abs(sps/(4*alpha))) < eps
        b[idx2] = alpha/(sps*np.sqrt(2))*((1+2/torch.pi)*np.sin(torch.pi/(4*alpha))+(1-2/torch.pi)*np.cos(torch.pi/(4*alpha)))

    return b


def generate_bins(sampling_freq, nSamples, device):
    """
    Generates the frequency and time bins for each signal. 

    Parameters:
        sampling_freq: int
            Sampling frequency for the signal.
        nSamples: int
            Total number of samples in the signal
    
    Returns:
        freq_bins: torch.tensor, shape (nSamples, )
        time_bins: torch.tensor, shape (nSamples, )
    """
    freq_bins = torch.fft.fftfreq(nSamples).to(device=device) * sampling_freq
    time_bins = torch.arange(nSamples).to(device=device) * 1/sampling_freq
    
    return freq_bins, time_bins


if __name__ == "__main__":
    sys.path.append("./")
    from datagen import SignalGenerator

    sps = 4
    alpha = 0.01
    pulse_dur = 401
    nSymbs = 128
    sampling_freq = 12e10
    freq_sep = 25e9
    M = 4
    nModes = 12

    sg = SignalGenerator(sps, alpha, pulse_dur, nSymbs, nModes, 1, freq_sep, sampling_freq, M)
    symbs = sg.generate_symbs(batch_size=10)
    sigs = sg.generate_signal(symbs)
    print(sigs.shape)
    print(get_power(sigs, dim=1, mode_dim=2))
    sigs = normalize(sigs, dim=1)
    print(get_power(sigs, dim=1, mode_dim=2))
    sigs = set_power(sigs, 10, dim=1, mode_dim=2)
    print(get_power(sigs, dim=1, mode_dim=2))
    print(get_power(sigs, dim=1, mode_dim=2))
    print(get_osnr(sigs.clone(), set_power(sigs, -10, dim=1, mode_dim=2), sampling_freq, dim=1, mode_dim=2))