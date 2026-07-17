import numpy as np
import matplotlib.pyplot as plt

# Define sample range
n = np.arange(-10, 11)

# 1. Unit Impulse Sequence
impulse = np.where(n == 0, 1, 0)

# 2. Unit Step Sequence
step = np.where(n >= 0, 1, 0)

# 3. Ramp Sequence
ramp = np.where(n >= 0, n, 0)

# 4. Exponential Sequence
a = 0.8
exponential = np.where(n >= 0, a**n, 0)

# 5. Sinusoidal Sequence
f = 0.1  # Frequency
sinusoidal = np.sin(2 * np.pi * f * n)

# Create subplots
plt.figure(figsize=(10, 12))

# Unit Impulse
plt.subplot(5, 1, 1)
plt.stem(n, impulse)
plt.title("Unit Impulse Sequence")
plt.xlabel("n")
plt.ylabel("δ(n)")
plt.grid(True)

# Unit Step
plt.subplot(5, 1, 2)
plt.stem(n, step)
plt.title("Unit Step Sequence")
plt.xlabel("n")
plt.ylabel("u(n)")
plt.grid(True)

# Ramp
plt.subplot(5, 1, 3)
plt.stem(n, ramp)
plt.title("Ramp Sequence")
plt.xlabel("n")
plt.ylabel("r(n)")
plt.grid(True)

# Exponential
plt.subplot(5, 1, 4)
plt.stem(n, exponential)
plt.title("Exponential Sequence (a = 0.8)")
plt.xlabel("n")
plt.ylabel("x(n)")
plt.grid(True)

# Sinusoidal
plt.subplot(5, 1, 5)
plt.stem(n, sinusoidal)
plt.title("Discrete-Time Sinusoidal Sequence")
plt.xlabel("n")
plt.ylabel("x(n)")
plt.grid(True)

plt.tight_layout()
plt.show()