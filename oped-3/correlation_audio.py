import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# ---------------------------------------------
# Function to Load Audio
# ---------------------------------------------
def load_audio(path):
    signal, sr = librosa.load(path, sr=22050, mono=True)
    return signal, sr


# ---------------------------------------------
# Normalize Audio
# ---------------------------------------------
def normalize(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


# ---------------------------------------------
# Pearson Correlation
# ---------------------------------------------
def pearson_corr(x, y):
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    corr = np.corrcoef(x, y)[0, 1]
    return corr


# ---------------------------------------------
# Cross Correlation
# ---------------------------------------------
def cross_corr(x, y):

    min_len = min(len(x), len(y))

    x = x[:min_len]
    y = y[:min_len]

    corr = correlate(x, y, mode='full')

    lag = np.argmax(corr) - (len(x)-1)

    max_corr = np.max(corr)

    return corr, lag, max_corr


# ---------------------------------------------
# Load Audio Files
# ---------------------------------------------
original, sr = load_audio("original_song.mp3")

karaoke, sr = load_audio("karaoke_song.mp3")

different, sr = load_audio("different_song.mp3")


# ---------------------------------------------
# Normalize
# ---------------------------------------------
original = normalize(original)
karaoke = normalize(karaoke)
different = normalize(different)


# ---------------------------------------------
# Pearson Correlations
# ---------------------------------------------
corr_ok = pearson_corr(original, karaoke)

corr_od = pearson_corr(original, different)

corr_kd = pearson_corr(karaoke, different)


print("--------------- Pearson Correlation ----------------")
print(f"Original vs Karaoke  : {corr_ok:.4f}")
print(f"Original vs Different: {corr_od:.4f}")
print(f"Karaoke vs Different : {corr_kd:.4f}")


# ---------------------------------------------
# Cross Correlation
# ---------------------------------------------
cross_ok, lag_ok, max_ok = cross_corr(original, karaoke)

cross_od, lag_od, max_od = cross_corr(original, different)

cross_kd, lag_kd, max_kd = cross_corr(karaoke, different)


print("\n--------------- Cross Correlation ----------------")
print(f"Original vs Karaoke")
print("Maximum Correlation :", max_ok)
print("Lag :", lag_ok)

print("\nOriginal vs Different")
print("Maximum Correlation :", max_od)
print("Lag :", lag_od)

print("\nKaraoke vs Different")
print("Maximum Correlation :", max_kd)
print("Lag :", lag_kd)


# ---------------------------------------------
# Plot Waveforms
# ---------------------------------------------
plt.figure(figsize=(15,8))

plt.subplot(3,1,1)
plt.plot(original)
plt.title("Original Song")

plt.subplot(3,1,2)
plt.plot(karaoke)
plt.title("Karaoke Version")

plt.subplot(3,1,3)
plt.plot(different)
plt.title("Different Song")

plt.tight_layout()
plt.show()


# ---------------------------------------------
# Plot Cross Correlation
# ---------------------------------------------
plt.figure(figsize=(15,10))

plt.subplot(3,1,1)
plt.plot(cross_ok)
plt.title("Cross Correlation: Original vs Karaoke")

plt.subplot(3,1,2)
plt.plot(cross_od)
plt.title("Cross Correlation: Original vs Different")

plt.subplot(3,1,3)
plt.plot(cross_kd)
plt.title("Cross Correlation: Karaoke vs Different")

plt.tight_layout()
plt.show()


# ---------------------------------------------
# Correlation Matrix
# ---------------------------------------------
matrix = np.array([
    [1, corr_ok, corr_od],
    [corr_ok, 1, corr_kd],
    [corr_od, corr_kd, 1]
])

labels = ["Original","Karaoke","Different"]

plt.figure(figsize=(6,5))

plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)

plt.xticks(range(3), labels)
plt.yticks(range(3), labels)

plt.colorbar(label="Correlation")

plt.title("Correlation Matrix")

for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{matrix[i,j]:.2f}",
                 ha='center',
                 va='center',
                 color='black',
                 fontsize=12)

plt.show()