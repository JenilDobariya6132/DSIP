from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os

# Change this to the actual location of your MP3 file
mp3_file = r"C:\Users\JENIL\Downloads\Haan Kar De.mp3"

# Check if file exists
if not os.path.exists(mp3_file):
    print("Error: MP3 file not found!")
    print("Current path:", mp3_file)
    exit()

# Load MP3
audio = AudioSegment.from_mp3(mp3_file)

# Convert to mono
audio = audio.set_channels(1)

# Extract samples
samples = np.array(audio.get_array_of_samples()).astype(np.float32)

# Normalize
samples = samples / np.max(np.abs(samples))

# Convolution kernel
kernel = np.array([1, 0, 1, 0, 1], dtype=np.float32)

# Perform convolution
convoluted = convolve(samples, kernel, mode="same")

# Normalize output
convoluted = convoluted / np.max(np.abs(convoluted))

# Convert back to int16
convoluted_int16 = (convoluted * 32767).astype(np.int16)

# Create output audio
convoluted_audio = AudioSegment(
    convoluted_int16.tobytes(),
    frame_rate=audio.frame_rate,
    sample_width=2,
    channels=1
)

# Save output
convoluted_audio.export("output_convoluted.wav", format="wav")

print("Convoluted audio saved as output_convoluted.wav")

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(samples)
plt.title("Original Audio Signal")

plt.subplot(2, 1, 2)
plt.plot(convoluted)
plt.title("Convoluted Audio Signal")

plt.tight_layout()
plt.show()