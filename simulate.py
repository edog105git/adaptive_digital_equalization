# Last Update: 2020.01.13
import numpy as np
import matplotlib.pyplot as plt
from PRBS import *
from scipy import signal

# Main file for simulating optical fibre transmission and detection (digital coherent) of a PM QPSK signal.
# 0. DEFINE VARIABLES
# ********** 0a. Constants ********** #
j = np.complex(0, 1)
# *********************************** #

# ********** 0b. Symbol stream parameters ********** #
n_prbs = 15                          # Number of bits in PRBS shift register
tap_ind = 3                          # Tap index for PRBS generator
n_sequence = np.power(2, n_prbs) - 1 # Number of symbols
# ************************************************** #

# ********** 0c. Time and frequency vectors ********** #
n_step = 24                                          # Number of time points per symbol
n_total = n_sequence * n_step                        # Total number of time points in symbol stream
B = 103e9                                            # Bit rate (bits/second)
FEC_overhead = 15                                    # Percent of FEC overhead (%)
B_coded = B*(1+(FEC_overhead/100))                   # Coded bit rate (bits/second)
Rs = B_coded/4                                       # Symbol rate (symbols/second) (2 polarizations + 2 quadratures)
Ts = 1/Rs                                            # Symbol interval (seconds)
t_step = Ts/n_step                                   # Size of time step in symbol (seconds)
t_vec = np.arange(0, n_total*t_step, t_step)         # Time vector (seconds)
f_s = n_step/Ts                                      # Sampling frequency (Hz)
omega_vec1 = np.linspace(0, np.pi*f_s, n_total//2)   # First half of frequency vector from 0 to pi*f_s (rad/s)
omega_vec2 = np.linspace(-np.pi*f_s, 0, n_total//2)  # Second half of frequency vector from -pi*f_s to 0 (rad/s)
omega_vec = np.concatenate((omega_vec1, omega_vec2)) # Frequency vector from 0 to pi*f_s + -pi*f_s to 0 (rad/s)
# **************************************************** #

# ********** 0d. Transmitter parameters ********** #
V_pi = 4              # Modulator half-wave voltage (V)
Tx_order = 5          # Order of Tx Bessel filter
Ptx = 1e-3            # Total transmitted power in 2 polarizations (W)
Tx_bandwidth = 0.8/Ts # Tx Bessel filter 3dB bandwidth (Hz)
# ************************************************ #




# 1. COMPUTE TRANSMITTED SYMBOL STREAM
# ********** 1a. Create symbol stream (2 quadratures and 2 polarizations) ********** #
bitstream1_i = PRBS(n_prbs, tap_ind, n_sequence) # Values 0 and 1
bitstream1_q = PRBS(n_prbs, tap_ind, n_sequence) # Values 0 and 1
bitstream2_i = PRBS(n_prbs, tap_ind, n_sequence) # Values 0 and 1
bitstream2_q = PRBS(n_prbs, tap_ind, n_sequence) # Values 0 and 1

x1_i = 2*bitstream1_i - 1 # Values -1 and 1
x1_q = 2*bitstream1_q - 1 # Values -1 and 1
x2_i = 2*bitstream2_i - 1 # Values -1 and 1
x2_q = 2*bitstream2_q - 1 # Values -1 and 1
# ********************************************************************************** #

# ********** 1b. Make rectangular drive waveforms ********** #
drive1_i = np.repeat(x1_i, n_step)*V_pi # Values -V_pi and V_pi
drive1_q = np.repeat(x1_q, n_step)*V_pi # Values -V_pi and V_pi
drive2_i = np.repeat(x2_i, n_step)*V_pi # Values -V_pi and V_pi
drive2_q = np.repeat(x2_q, n_step)*V_pi # Values -V_pi and V_pi
# ********************************************************** #

# ********** 1c. Filter drive waveforms to obtain composite response of transmitter ********** #
# NOTE: Assume drive signals of modulator produced by passing rectangular pulse train through lowpass Bessel filter with order Tx_order and 3-dB bandwidth of Tx_bandwidth.
b, a = signal.bessel(Tx_order, 2*np.pi*Tx_bandwidth, analog=True, norm='mag')
omega_tx, h_tx = signal.freqs(b, a, worN=omega_vec)

# Remove phase shift response
gd_tx = -np.diff(np.unwrap(np.angle(h_tx)))/(omega_tx[1]-omega_tx[0])
h_tx = h_tx*np.exp(j*omega_tx*gd_tx[0])

filtered_d1_i = np.real(np.fft.ifft(np.fft.fft(drive1_i)*h_tx))
filtered_d1_q = np.real(np.fft.ifft(np.fft.fft(drive1_q)*h_tx))
filtered_d2_i = np.real(np.fft.ifft(np.fft.fft(drive2_i)*h_tx))
filtered_d2_q = np.real(np.fft.ifft(np.fft.fft(drive2_q)*h_tx))
# ******************************************************************************************** #

# ********** 1d. Set up CW response of laser ********** #
E_carrier = np.ones(np.size(t_vec))
# ***************************************************** #

# ********** 1e. Modulate CW laser output with drive waveforms ********** #
E1_i = E_carrier * np.sin((np.pi*filtered_d1_i)/(V_pi*2)) # Field is -1 when drive signal at -V_pi and 1 when drive signal at V_pi
E1_q = E_carrier * np.sin((np.pi*filtered_d1_q)/(V_pi*2))
E2_i = E_carrier * np.sin((np.pi*filtered_d2_i)/(V_pi*2))
E2_q = E_carrier * np.sin((np.pi*filtered_d2_q)/(V_pi*2))

E1 = E1_i + (j*E1_q)                                      # Field vector for x-polarization
E2 = E2_i + (j*E2_q)                                      # Field vector for y-polarization
# *********************************************************************** #

# ********** 1f. Pulse carving and scalling to correct power ********** #
# NOTE: Assume using NRZ modulation format.
E1_carved = 1.0720 * E1
E2_carved = 1.0720 * E2

E1_tx = (np.sqrt(Ptx)/2)*E1_carved
E2_tx = (np.sqrt(Ptx)/2)*E2_carved
# ********************************************************************* #

# ********** FOR TESTING PURPOSES ********** #
plt.figure()
plt.stem(np.real(E1[0:n_step*8]))
plt.show()

plt.figure()
plt.stem(np.imag(E1[0:n_step*8]))
plt.show()

plt.figure()
plt.stem(np.real(E2[0:n_step*8]))
plt.show()

plt.figure()
plt.stem(np.imag(E2[0:n_step*8]))
plt.show()
# ****************************************** #
