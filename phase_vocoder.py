# Based on https://github.com/KAIST-MACLab/PyTSMod/blob/main/pytsmod/utils/stft.py
# So I guess GPL-3 license applies
# Mostly simplified the code to handle only integer scale factor and improve readability.

import numpy as np

# Creates curve, that somehow properly mixes the volume.
# used for post processing: signal_out = signal_out / cutoff_curve_get(win_size, steps, syn_step) 
# It has nice ramp-up/down feature, but seems that it introduces a tiny bit of HF component to signal.
# Currently unused. Steady state value is ~1.5.

#def cutoff_curve_get(win_size, n_frames, syn_hop):
#    curve = np.zeros(n_frames * syn_hop  + win_size)
#    for i in range(n_frames):
#        curve[win_pos: win_pos + win_size] += np.power(np.hanning(win_size), 2) 
#    curve[curve < 1e-3] = 1e-3
#    return curve

def fft_rephase(fft, fft_prev, fft_rephased_last, win_size, anal_step, syn_step):
    k = np.arange(win_size)             #const
    omega = 2 * np.pi * k / win_size    #const
    phase_curr = np.angle(fft)
    phase_last = np.angle(fft_prev)
    dphi = omega * anal_step
    hpi = (phase_curr - phase_last) - dphi
    hpi = hpi - 2 * np.pi * np.round(hpi / (2 * np.pi))   #??? 
    ipa_hop = (omega + hpi / anal_step) * syn_step
    phase_syn = np.angle(fft_rephased_last)
    theta = phase_syn + ipa_hop - phase_curr
    phasor = np.exp(1j * theta)
    return phasor * fft

def phase_vocoder(signal, fac, win_size=2048, syn_step=512):
    anal_range = signal.shape[0] + win_size // (2 * fac)
    anal_step = int(syn_step / fac)
    steps = np.math.ceil(anal_range / anal_step)

    output_length = int(fac * signal.shape[0])
    signal_out = np.zeros(syn_step * steps + win_size)
    sig_padded = np.pad(signal, (win_size // 2, win_size + anal_range), 'constant')

    fft_last = np.zeros(win_size) # Reference buffers.
    rephased_fft_last = np.zeros(win_size)

    for step in range(0, steps):
        sig_pos = anal_step * step
        fft = np.fft.fft(sig_padded[sig_pos: sig_pos + win_size] * np.hanning(win_size))
        
        if step == 0:
            fft_rephased = fft  # First chunk can not be rephased, but it is used for phase initialization.
        else:
            fft_rephased = fft_rephase(fft, fft_last, rephased_fft_last, win_size, anal_step, syn_step)
        fft_last = fft
        rephased_fft_last = fft_rephased

        win_pos = syn_step * step
        signal_out[win_pos: win_pos + win_size] += np.real(np.fft.ifft(fft_rephased)) * np.hanning(win_size) / 1.5 # Normalize volume after mixing hanning windows.

    signal_out = signal_out[win_size // 2: - win_size // 2]
    signal_out = signal_out[0 : output_length]
    return signal_out.squeeze()


if __name__ == "__main__":
    import soundfile as sf
    x, sr = sf.read('/home/me/Desktop/t3.wav')
    out = phase_vocoder(x, 2)
    sf.write('/home/me/Desktop/out.wav', out.T, sr)
