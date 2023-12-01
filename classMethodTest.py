# import itertools
# import numpy as np
# from matplotlib import pyplot as plt
# class A:
#     def __init__(self, amt) -> None:
#         self.gen = (np.sin(k)*amt for k in itertools.count(0,0.1))


#     def Output(self):
#         yield next(self.gen)
    
# class B:
#     def __init__(self) -> None:
#         self.data = []
    
#     def Input(self, info):
#         self.data.append(next(info))
#         print(self.data)

# a = A(10)
# amethod = a.Output
# b = B()
# bmethod = b.Input
# c = A(400)
# c.gen = (np.sin(k)*2 for k in itertools.count(0,0.1))
# c.gen = a.gen


# bmethod(amethod())
# bmethod(amethod())

# print(c == a)
# print(c.gen== a.gen, c.gen, a.gen)
# print(c.gen.gi_code == a.gen.gi_code)

# def sinwave():
#     return (np.sin(i*(2*np.pi*440)/44100)*2 for i in itertools.count(0,1))

# s = sinwave()
# signal = np.array([next(s) for i in range(256*4)])
 
# fc = 440/44100  # Cutoff frequency as a fraction of the sampling rate (in (0, 44100)).
# b = 300/44100# Transition band -- the width we want the band to cover as rolloff, units WLOG f_c
# N = int(np.ceil((4 / b)))
# if not N % 2: N += 1  # Make sure that N is odd.
# n = np.arange(N)
 
# # Compute sinc filter.
# h = np.sinc(2 * fc * (n - (N - 1) / 2))

# # Compute Blackman window.
# w = np.blackman(N)
 
# # Multiply sinc filter by window.
# k = h * w
# #normalize 
# k /= np.sum(k)
# #now make it between 18 and -18 db

# plt.plot(n,k, 'r-') # sinc and window multiplied together
# plt.show()

# def fftPlot(s, dt=1/44100, plot=True):
#         # https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
#         # Here it's assumes analytic signal (real signal...) - so only half of the axis is required
#         if dt is None:
#             dt = 1
#             t = np.arange(0, s.shape[-1])
#             xLabel = 'samples'
#         else:
#             t = np.arange(0, s.shape[-1]) * dt
#             xLabel = 'freq [Hz]'

#         if s.shape[0] % 2 != 0:
#             t = t[0:-1]
#             s = s[0:-1]

#         sigFFT = np.fft.fft(s) / t.shape[0]  # Divided by size t for coherent magnitude

#         freq = np.fft.fftfreq(t.shape[0], d=dt)

#         # Plot analytic signal - right half of frequence axis needed only...
#         firstNegInd = np.argmax(freq < 0)
#         freqAxisPos = freq[0:firstNegInd]
#         sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal
#         # normalizing
#         sigFFTPos = 2*(sigFFTPos-np.min(sigFFTPos))/(np.max(sigFFTPos) - np.min(sigFFTPos)) - 1

#         if plot:
#             plt.plot(freqAxisPos, np.abs(sigFFTPos))
#             plt.xlabel(xLabel)
#             plt.ylabel('mag')
#             plt.title('Analytic FFT plot')
#             plt.xscale('log')
#             plt.show()

#         return sigFFTPos, freqAxisPos

# fftPlot(k)
# print(k.size)

import numpy as np
import matplotlib.pyplot as plt

# def fir_lowpass_filter(input_signal, cutoff_freq, num_taps):
#     # Design FIR filter coefficients
#     nyquist = 0.5
#     taps = np.sinc(2 * cutoff_freq * (np.arange(num_taps) - (num_taps - 1) / 2.)) / (np.pi * (nyquist / 2))

    
#     # Apply the filter to the input signal using convolution
#     output_signal = np.convolve(input_signal, taps, mode='same')
#     fftPlot(output_signal)
    
#     return output_signal

# # Generate a sample signal
# fs = 1000  # Sampling frequency
# t = np.arange(0, 1, 1/fs)  # Time vector
# input_signal = np.sin(2 * np.pi * 440 * t) + 0.5 # * np.sin(2 * np.pi * 120 * t)

# # Design and apply the FIR low-pass filter
# cutoff_frequency = 200  # Adjust as needed
# num_filter_taps = 51  # Adjust as needed

# output_signal = fir_lowpass_filter(input_signal, cutoff_frequency / fs, num_filter_taps)

# # Plot the results
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(t, input_signal, label='Input Signal')
# plt.title('Original Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(t, output_signal, label='Filtered Signal', color='orange')
# plt.title('Filtered Signal using FIR Low-pass Filter')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.tight_layout()
# plt.show()

data= np.fromfile("./s.txt")
plt.plot(data)
plt.show()