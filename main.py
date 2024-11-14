import numpy as np                  # for working with arrays, fourier transforms, linear algebra and matrices
import matplotlib.pyplot as plt     # for visualization, to plot the data
import scipy.interpolate            # sub-package for objects used in interpolation
from itertools import product       # to easily iterate the mapping table


'''
   I use my explanation of OFDM found in: 
   https://github.com/TomSimkin/Topics/blob/main/OFDM%20(Orthogonal%20Frequency-Division%20Multiplexing).docx
   Pay attention to the OFDM Modem diagram, I will code the system according to the diagram.
   Note - I did not code the P/S, D/A, A/D and S/P (at Rx) simply because this simulation does not incorporate hardware.
   If you wish to use hardware, add the relevant libraries and the missing components to the code.
   Feel free to experiment and adjust any value to see how the system changes!
'''

'''
1.  Define the basic parameters of the OFDM: pilots, data and cyclic prefix.
'''
subCarriers = 64
cyclicPrefix = subCarriers // 4 # 25%
pilot = 8
pilotVal = 3+3j                 # amp = 4.24, phase angle = pi/4 radians

allCarriers = np.arange(subCarriers)                # indices of all subcarriers ([0, 1, ... subCarriers - 1])
pilotCarriers = allCarriers[::subCarriers//pilot]   # place pilot in every (subcarriers/pilot)th carrier

pilotCarriers = np.append(pilotCarriers, allCarriers[-1]) # adding last subcarrier as a pilot (for channel estimation)
pilot = pilot + 1                                         # update total num of pilots

dataCarriers = np.delete(allCarriers, pilotCarriers) # exclude the pilot carriers from the data carriers
'''
2.  Plot the carriers.
'''
plt.figure(figsize=(8,1.5))
plt.title("Carriers")
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label = 'pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label = 'data')
plt.legend(fontsize = 10, ncol = 2)
plt.xlim((-1, subCarriers))
plt.ylim((-0.1, 0.3))
plt.yticks([])
plt.tight_layout()
plt.grid(True)
plt.savefig("pilots.png")
'''
3.  Plot the constellation. For this OFDM I will use 16QAM.
    μ = 4 (num of bits per symbol = log2(num of constellation points))
'''
mu = 4
totalData_per_symbol = len(dataCarriers) * mu # number of bits that can be transmitted in a single OFDM symbol (224)
mapping_table = {                             # map all the bits with gray-mapping, ensuring that points differ by
    (0,0,0,0) : -3-3j,                        # only 1 bit are adjacent to each other in the constellation.
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

plt.figure(figsize=(5,5))
for B in product([0,1], repeat = 4):
    Q = mapping_table[B]
    plt.plot(Q.real, Q.imag, 'bo')
    plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha = 'center')

plt.title("16-QAM Constellation with Gray-Mapping")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.axis('equal')
plt.xlim(-4 ,4)
plt.ylim(-4, 4)
plt.tight_layout()
plt.grid(True)
plt.savefig("16qam_constellation_map.png")
'''
4.  Define the wireless channel between the Tx and Rx.
    To enhance the simulation's realism I will define the channel response and add SNR.
'''
channelResponse = np.array([1, 0, 0.3+0.3j]) # the impulse response of the wireless channel:
                                             # 1'st element - main signal path
                                             # 2'nd element - no signal received
                                             # 3'rd element - delayed path, amp - 0.424, phase shift - pi/4 radians
FFT_channelResponse = np.fft.fft(channelResponse, subCarriers) # FFT to channel response for channel estimation
SNR = 24 # signal-to-noise ratio in dB
'''
5. Plot the channel response
'''
plt.figure(figsize=(6,4))
plt.title("Channel Response")
plt.plot(allCarriers, abs(FFT_channelResponse))
plt.xlabel("Subcarrier index")
plt.ylabel("$|H(f)|$")
plt.xlim(0, subCarriers)
plt.ylim(0.4, 1.6)
plt.tight_layout()
plt.grid(True)
plt.savefig("channel_response.png")
'''
6. Begin the OFDM process, define the bits entering the modem and S/P.
'''
bits = np.random.binomial(n=1, p=0.5, size=totalData_per_symbol) # generate binary data using Bernoulli distribution
                                                                 # total num of data bits (excluding pilots) = 220
def s_to_p(bits):
    return bits.reshape((len(dataCarriers), mu))

bits_SP = s_to_p(bits)
'''
7. Encode the parallel bits using the mapping table from step 3.
   Also, allocate each subcarrier with data/pilot - create OFDM symbol
'''
def encoder(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])    # convert bit group according to the mapping table

qam_VAL = encoder(bits_SP)

def create_symbol(data):
    symbol = np.zeros(subCarriers, dtype=complex)
    symbol[pilotCarriers] = pilotVal
    symbol[dataCarriers] = data
    return symbol

data_OFDM = create_symbol(qam_VAL)
'''
8. Transform the symbol to the time-domain using IFFT
'''
def ifft(data_OFDM):
    return np.fft.ifft(data_OFDM)

time_OFDM = ifft(data_OFDM)
''' 
9. Add cyclic prefix to the symbol
'''
def add_cp(time_OFDM):
    cp = time_OFDM[-cyclicPrefix:]
    return np.append(cp, time_OFDM)

cp_OFDM = add_cp(time_OFDM)
'''
10. Define the wireless channel as a static multipath channel with impulse response
'''
def channel(signal):
    convolved = np.convolve(signal, channelResponse) # the signal at Rx is the convolution of the transmit signal
                                                     # with the channel response
    p_avg= np.mean(abs(convolved**2))            # no need to divide by 2 as the signal is complex-valued
    noise_power = p_avg * 10**(-SNR/10)          # noise power = σ^2
                                                 # using SNR ratio instead of N=kTB as the latter is less practical
    print("RX Signal power: %.4f. Noise power: %.4f" % (p_avg, noise_power))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    # noise explanation:
    # np.sqrt(sigma2/2) - noise_power, split equally between real and imaginary parts
    # np.random.randn(*convolved.shape) - generates real-valued Gaussian (normal) random numbers.
    # Mean is 0, and variance is 1.
    # *convolved.shape unpacks the shape of the convolved signal, ensuring the noise has the same dimensions.
    # 1j*np.random.randn(*convolved.shape) - generates the imaginary part of the noise.
    # It's another set of Gaussian random numbers, multiplied by 1j to make them imaginary.
    return convolved + noise
'''
11. Plot the transmit and receive signal
'''
transmit_sig = cp_OFDM
receive_sig = channel(transmit_sig)

plt.figure(figsize=(8,3))
plt.title("OFDM Signals")
plt.plot(abs(transmit_sig), label='Tx signal')
plt.plot(abs(receive_sig), label='Rx signal')
plt.legend(fontsize=10)
plt.xlabel('Time')  # the x-axis represents one sample of the OFDM signal, the time depends on the sampling rate
plt.ylabel('$|signal|$')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.grid(True)
plt.savefig("signals.png")
'''
12. Remove the CP at Rx 
'''
def remove_cp(signal):
    return signal[cyclicPrefix:(cyclicPrefix+subCarriers)]

receive_NOCP= remove_cp(receive_sig)
'''
13. Transform the symbol back to the frequency-domain using FFT
'''
def fft(recieve_NOCP):
    return np.fft.fft(recieve_NOCP)

freq_OFDM = fft(receive_NOCP)
'''
14. Invert channel (frequency domain equalizer). 
    Channel estimation using zero-forcing (to mitigate ISI and co-channel interference)
    followed by linear interpolation.
    Also, plot the channel estimation.
'''
def channel_estimate(freq_OFDM):
    pilots = freq_OFDM[pilotCarriers]
    pilot_estimates = pilots / pilotVal # By dividing the received pilot values by the known transmitted pilot values,
                                        # we get an estimate of the channel's effect on these subcarriers.
    channelEstimation_amp = np.interp(allCarriers, pilotCarriers, np.abs(pilot_estimates))     # absolute val
    channelEstimation_phase = np.interp(allCarriers, pilotCarriers, np.angle(pilot_estimates)) # phase
    full_estimate = channelEstimation_amp * np.exp(1j*channelEstimation_phase) # polar form

    plt.figure(figsize=(12,6))
    plt.title("Channel Estimation")
    plt.plot(allCarriers, abs(FFT_channelResponse), label='Correct channel')
    plt.stem(pilotCarriers, abs(pilot_estimates), label='Pilot estimates')
    plt.plot(allCarriers, abs(full_estimate), label='Estimated channel interpolation')
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Carrier index')
    plt.ylabel('$|H(f)|$')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("channel_estimation.png")

    return full_estimate

estimated_signal = channel_estimate(freq_OFDM)

def domain_equalizer(freq_OFDM, estimated_signal, threshold):
    mask = np.abs(estimated_signal) < threshold # boolean mask identifying channel estimates below the threshold
    safe = np.where(mask, np.sign(estimated_signal) * threshold, estimated_signal) # replaces small values with the
                                                                                   # threshold value
    equalized = freq_OFDM / safe # the noise is not factored in the equalization for simplicity
    equalized[mask] = 0          # equalized values set to zero for subcarriers with very weak channel estimates,
                                 # can help prevent noise amplification.
    return equalized

threshold = np.mean(np.abs(estimated_signal)**2) * 0.01 # 1% of average channel
equalized_signal = domain_equalizer(freq_OFDM, estimated_signal, threshold)

'''
15. Extract the data carriers from the equalized symbol.
    Also, plot the data.
    Note - this step is not shown in the diagram.
'''
def get_data(equalized_signal):
    return equalized_signal[dataCarriers]

estimated_data = get_data(equalized_signal)

plt.figure(figsize=(5,5))
plt.title("16-QAM Received Constellation")
plt.plot(estimated_data.real, estimated_data.imag, 'bo')
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.axis('equal')
plt.xlim(-4 ,4)
plt.ylim(-4, 4)
plt.tight_layout()
plt.grid(True)
plt.savefig("16qam_received_constellation.png")
'''
16. Decode the data. In order to do this, we compare each received constellation point against each possible 
    constellation point and choose the constellation point which is closest to the received point.
'''
decoding_table = {v : k for k, v in mapping_table.items()} # inverse mapping of the mapping table from step 3

def decoder(estimated_data):
    known_constellation = np.array([x for x in decoding_table.keys()])
    distance = abs(known_constellation.reshape(1,-1) - estimated_data.reshape(-1, 1)) # after reshaping, distance 2D:
                                                                                      # known_constellation (1, n)
                                                                                      # estimated_data (m,1)
    closest_index = distance.argmin(axis=1)
    closest_points = known_constellation[closest_index]

    return np.vstack([decoding_table[C] for C in closest_points]), closest_points # returns 2D array of closest_point bits
                                                                                # and the closest_point array
decoder_bits, closest_points = decoder(estimated_data)
'''
17. Plot the points
'''
plt.figure(figsize=(5,5))
for estimate, closest in zip(estimated_data, closest_points):
    plt.plot([estimate.real, closest.real], [estimate.imag, closest.imag], 'b-o')
    plt.plot(closest_points.real, closest_points.imag, 'ro')

plt.title("16-QAM Constellation with Received Symbols")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.axis('equal')
plt.xlim(-4 ,4)
plt.ylim(-4, 4)
plt.tight_layout()
plt.grid(True)
plt.savefig("16qam_constellation_decoding.png")
'''
18. Convert the decoded bits from parallel to series (P/S) and calculate the bit error rate
'''
def p_to_s(bits):
    return bits.reshape(-1,)

final_bits = p_to_s(decoder_bits)
print("Bit error rate:", np.sum(abs(bits-final_bits)/len(bits))) # high SNR = lower chance of bit errors
