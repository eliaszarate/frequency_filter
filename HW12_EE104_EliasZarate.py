#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################
#   Name: Elias Zarate
#   Date: 11/22/2020
#   HW: 11
#   
#########################################################

from scipy.io.wavfile import write
import numpy as np 
import matplotlib.pyplot as plt
import numpy
from scipy import fftpack
from scipy import signal
from scipy.io import wavfile
import pandas as pd

######################## Part 1 #########################

    
output_path = "./"

def func(frequency, wavelength, time): 
    f = frequency
    w = np.pi * 2 * f 
    t = time 
    return np.sin(w*t)

time = np.linspace(0, 20, 10000)

# Frequncy of each note
frequency_C_sharp = 17.01 # C#0/Db0 Hz
frequency_D_flat = 38.18 #  D#1/Eb1 Hz
frequency_B7 = 23879.23 # B7 Hz

# Wavelength of each note
wavlength_C_sharp = 2028.35 # C#0/Db0 cm
wavlength_D_flat = 903.53 #  D#1/Eb1 cm
wavlength_B7 = 8.89 # B7 cm

# Obtaining Data
C_sharp = func(frequency_C_sharp,wavlength_C_sharp, time)
D_flat = func(frequency_D_flat,wavlength_D_flat, time )
B7 = func(frequency_B7,wavlength_B7, time)

# Sum of wave functions 
wave_sum = C_sharp + D_flat + B7

# Plot data
plt.title('Wave functions')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.xlim(0,50)
plt.plot(B7)
plt.plot(D_flat)
plt.plot(C_sharp)
plt.show()

# Sum of wave functions 
plt.title('Sum of Wave functions')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.xlim(0,20)
plt.plot(time, wave_sum)
plt.show()

# Save values into csv file
file= np.asarray(C_sharp)
np.savetxt("signals.csv", file, delimiter=" ")


#print(type(time))
#print(type(C_sharp))

# Writing a numpy array to WAV file
samplerate = 44100; fs = 100
t = np.linspace(0., 1., samplerate)
amplitude = np.iinfo(np.int16).max
data = wave_sum
write("foo.wav", samplerate, data = wave_sum)
write("b7.wav", samplerate, data = B7)
write("Dflat.wav", samplerate, data = D_flat)
write("C#.wav", samplerate, data = C_sharp)


######################## Part 2 #########################

#s step response
time_step=0.001

# Retrun descrite Fourier transform
sig_fft = fftpack.fft(wave_sum)
power = np.abs(sig_fft)**2
sample_freq = fftpack.fftfreq(wave_sum.size, d = time_step)

# Plot frequency domain
plt.plot(sample_freq,power)
plt.title('Frequency Domain')
plt.ylabel("Power")
plt.xlabel("Frequency [Hz]")
plt.show()

pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]


high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(10,5))
plt.plot(time, wave_sum, label='Original signal')
plt.plot(time, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')

#sig = filtered_sig
# The FFT of the signal
sig_fft1 = fftpack.fft(filtered_sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft1)**2

# The corresponding frequencies
sample_freq = fftpack.fftfreq(filtered_sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(30, 20))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')
# It should look very clean now

t = time

b, a = signal.butter(3, 0.05)

# Apply the filter to sig. Use lfilter_zi to choose the initial condition of the filter:
zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, wave_sum, zi=zi*wave_sum[0])

# Apply the filter again, to have a result filtered at an order the same as filtfilt:
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

# Use filtfilt to apply the filter:
y = signal.filtfilt(b, a, C_sharp)


write("C#_copy_result.wav", samplerate, data = y)

print('\n\n\nSaving results from filtered sigal to "C#_copy_result.wav:"\n\n\n')

input_filename = 'C#_copy_result.wav'

samplerate, data = wavfile.read(str('./' + input_filename))
print('Load is Done! \n')

wavData = pd.DataFrame(data)

# From: https://github.com/Lukious/wav-to-csv/blob/master/wav2csv.py
# Separates if there are two channel
if len(wavData.columns) == 2:
    print('Stereo .wav file\n')
    wavData.columns = ['R', 'L']
    stereo_R = pd.DataFrame(wavData['R'])
    stereo_L = pd.DataFrame(wavData['L'])
    print('Saving...\n')
    stereo_R.to_csv(str(input_filename[:-4] + "_Output_stereo_R.csv"), mode='w')
    stereo_L.to_csv(str(input_filename[:-4] + "_Output_stereo_L.csv"), mode='w')
    print('Save is done, please see: ' + str(input_filename[:-4]) + '_Output_stereo_R.csv and '
                          + str(input_filename[:-4]) + '_Output_stereo_L.csv for output results\n')
    
elif len(wavData.columns) == 1:
    print('Mono .wav file\n')
    wavData.columns = ['M']
    stereo_uno = pd.DataFrame(wavData['M'])
    wavData.to_csv(str(input_filename[:-4] + "_Output_mono.csv"), mode='w')

    print('Save is done ' + str(input_filename[:-4]) + '_Output_mono.csv')

else:
    print('Multi channel .wav file\n')
    print('number of channel : ' + len(wavData.columns) + '\n')
    wavData.to_csv(str(input_filename[:-4] + "Output_multi_channel.csv"), mode='w')

    print('Save is done ' + str(input_filename[:-4]) + 'Output_multi_channel.csv')
