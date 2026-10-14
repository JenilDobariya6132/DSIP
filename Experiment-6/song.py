
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, freqz, lfilter

# Step 1: Load the song
Fs, song = wavfile.read("D:\DSIP\Experiment-6\😱😱Ma Pava te Gadha thi Utarua Maha Kali re. Gujrati Garba Song.Navratri Status 😍 - Virendrasinh Chauhan official (128k).wav")

# Convert stereo song to mono
if song.ndim == 2:
    song = song.mean(axis=1)

# Normalize song
song = song.astype(float)
song = song / (np.max(np.abs(song)) + 1e-10)

print("Sampling Frequency:", Fs)
print("Song Duration:", len(song) / Fs, "seconds")

# Step 2: Filter parameters
N = 31

# 6 kHz requires a sampling frequency greater than 12 kHz
if Fs <= 12000:
    raise ValueError(
        "Use a song with sampling frequency greater than 12000 Hz."
    )

# Step 3: Design four FIR filters
low = firwin(N, 3000, window="hann",
             pass_zero="lowpass", fs=Fs)

high = firwin(N, 3000, window="hann",
              pass_zero="highpass", fs=Fs)

band = firwin(N, [3000, 6000], window="hann",
              pass_zero="bandpass", fs=Fs)

stop = firwin(N, [3000, 6000], window="hann",
              pass_zero="bandstop", fs=Fs)

filters = [low, high, band, stop]

titles = ["Low-Pass Filter", "High-Pass Filter",
          "Band-Pass Filter", "Band-Stop Filter"]

# Step 4: Plot magnitude, impulse and phase responses
plt.figure(figsize=(12, 12))

for i in range(4):
    b = filters[i]

    f, H = freqz(b, worN=2048, fs=Fs)

    # Magnitude response
    magnitude = 20 * np.log10(np.maximum(np.abs(H), 1e-10))

    # Impulse response
    impulse = b

    # Phase response
    phase = np.unwrap(np.angle(H))

    # Plot magnitude
    plt.subplot(4, 3, i * 3 + 1)
    plt.plot(f, magnitude)
    plt.title(titles[i] + " - Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()

    # Plot impulse
    plt.subplot(4, 3, i * 3 + 2)
    plt.stem(np.arange(N), impulse)
    plt.title(titles[i] + " - Impulse")
    plt.xlabel("Sample (n)")
    plt.ylabel("Amplitude")
    plt.grid()

    # Plot phase
    plt.subplot(4, 3, i * 3 + 3)
    plt.plot(f, phase)
    plt.title(titles[i] + " - Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (Radians)")
    plt.grid()

plt.tight_layout()
plt.show()

# Step 5: Apply filters to the song
plt.figure(figsize=(12, 8))

for i in range(4):
    filtered_song = lfilter(filters[i], 1, song)

    # Display filtered song waveform
    plt.subplot(4, 1, i + 1)
    time = np.arange(len(song)) / Fs
    plt.plot(time, filtered_song)
    plt.title(titles[i] + " - Filtered Song")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid()

    # Save filtered song
    output = np.int16(
        filtered_song / (np.max(np.abs(filtered_song)) + 1e-10)
        * 32767
    )
    wavfile.write(
        ["low_pass.wav", "high_pass.wav",
         "band_pass.wav", "band_stop.wav"][i],
        Fs, output
    )

plt.tight_layout()
plt.show()

print("All four filtered songs have been saved!")
