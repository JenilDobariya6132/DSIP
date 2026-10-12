import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, bilinear, freqz, cheby1, lfilter
def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):
    analog_b, analog_a = butter(
        filter_order,
        cutoff_frequency,
        analog=True,
        btype='low'
    )
    digital_b, digital_a = bilinear(
        analog_b,
        analog_a,
        sampling_frequency
    )
    return digital_b, digital_a
def design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
):
    analog_b, analog_a = cheby1(
        filter_order,
        ripple,
        cutoff_frequency,
        analog=True,
        btype='low'
    )   
    # Perform bilinear transformation
    digital_b, digital_a = bilinear(
        analog_b,
        analog_a,
        sampling_frequency
    )
    return digital_b, digital_a
def plot_filter_response(digital_b, digital_a, sampling_frequency, filter_name):
    frequency, response = freqz(
        digital_b,
        digital_a,
        fs=sampling_frequency,
        worN=4096
    )
    magnitude = np.abs(response)
    plt.figure(figsize=(10, 6))
    plt.plot(frequency, magnitude)
    plt.title(filter_name + " - Magnitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.xlim(0, sampling_frequency / 2)
    plt.show()
    impulse = np.zeros(100)
    impulse[0] = 1
    impulse_response = lfilter(
        digital_b,
        digital_a,
        impulse
    )
    plt.figure(figsize=(10, 6))
    plt.stem(
        range(len(impulse_response)),
        impulse_response
    )
    plt.title(filter_name + " - Impulse Response")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
filter_order = 4
cutoff_frequency = 1000
sampling_frequency = 8000
ripple = 0.5
digital_b, digital_a = design_butterworth_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency
)
print("\n======================================")
print("BUTTERWORTH FILTER")
print("======================================")
print("Numerator coefficients (b):")
print(digital_b)
print("\nDenominator coefficients (a):")
print(digital_a)
plot_filter_response(
    digital_b,
    digital_a,
    sampling_frequency,
    "Butterworth Filter"
)
digital_b, digital_a = design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
)
print("\n======================================")
print("CHEBYSHEV TYPE-I FILTER")
print("======================================")
print("Numerator coefficients (b):")
print(digital_b)
print("\nDenominator coefficients (a):")
print(digital_a)
plot_filter_response(
    digital_b,
    digital_a,
    sampling_frequency,
    "Chebyshev Type-I Filter"
)
filter_path = "filter_coefficients.txt"
np.savetxt(
    filter_path,
    np.vstack((digital_b, digital_a)),
    delimiter=","
)
print("\nFilter coefficients saved at:", filter_path)   
        #generate a discrete sine wave of 1KHz,2Khz,3Khz and 4Khz plot all these
fs = 10000  # 10 kHz
duration = 0.005  
n = np.arange(0, duration, 1 / fs)
f1 = 1000   # 1 kHz
f2 = 2000   # 2 kHz
f3 = 3000   # 3 kHz
f4 = 4000   # 4 kHz
x1 = np.sin(2 * np.pi * f1 * n)
x2 = np.sin(2 * np.pi * f2 * n)
x3 = np.sin(2 * np.pi * f3 * n)
x4 = np.sin(2 * np.pi * f4 * n)
plt.figure(figsize=(10, 6))
plt.stem(n, x1)
plt.title("Discrete Sine Wave - 1 kHz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.stem(n, x2)
plt.title("Discrete Sine Wave - 2 kHz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.stem(n, x3)
plt.title("Discrete Sine Wave - 3 kHz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.stem(n, x4)
plt.title("Discrete Sine Wave - 4 kHz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
# add all these four signal and plot it in time as well as in frequency domain
combined_signal = x1 + x2 + x3 + x4
plt.figure(figsize=(10, 6))
plt.stem(n, combined_signal)
plt.title("combined signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
#frequency domain plot of combined signal
fs=10000
n = np.arange(0, duration, 1 / fs)
duration = 0.005
plt.figure(figsize=(10, 6))
plt.stem(n, combined_signal)
plt.title("Combined Signal - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
#pass through butterworth filter having cutoff frequency of 2KHz and plot the output in time as well as in frequency domain
cutoff_frequency = 2000  # 2 kHz
filter_order = 4
digital_b, digital_a = design_butterworth_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency
)
filtered_signal = lfilter(digital_b, digital_a, combined_signal)
# Plot filtered signal in time domain
plt.figure(figsize=(10, 6))
plt.stem(n, filtered_signal)
plt.title("Filtered Signal - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
# Plot filtered signal in frequency domain
filtered_fft = np.fft.rfft(filtered_signal)
positive_frequency = np.fft.rfftfreq(len(filtered_signal), d=1 / sampling_frequency)
plt.figure(figsize=(10, 6))
plt.plot(positive_frequency, np.abs(filtered_fft))
plt.title("Filtered Signal - Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
#pass through chebyshev filter having cutoff frequency of 2KHz and plot the output in time as well as in frequency domain
cutoff_frequency = 3000  # 3 kHz
filter_order = 4
digital_b, digital_a = design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
)
filtered_signal_chebyshev = lfilter(digital_b, digital_a, combined_signal)
# Plot filtered signal in time domain
plt.figure(figsize=(10, 6))
plt.stem(n, filtered_signal_chebyshev)
plt.title("Filtered Signal (Chebyshev) - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
# Plot filtered signal in frequency domain
filtered_fft_chebyshev = np.fft.rfft(filtered_signal_chebyshev)
positive_frequency_chebyshev = np.fft.rfftfreq(len(filtered_signal_chebyshev), d=1 / sampling_frequency)
plt.figure(figsize=(10, 6))
plt.plot(positive_frequency_chebyshev, np.abs(filtered_fft_chebyshev))
plt.title("Filtered Signal (Chebyshev) - Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
