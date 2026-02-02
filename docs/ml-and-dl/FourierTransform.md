# Fourier Transform

The Fourier Transform is a mathematical technique that decomposes a signal into its constituent frequencies. It transforms data from the **time/space domain** to the **frequency domain**, revealing hidden periodic patterns.

---

## Core Intuition

### The Big Idea

Any signal (audio, image, time series) can be represented as a **sum of sine and cosine waves** at different frequencies. The Fourier Transform tells you:
- **Which frequencies** are present in your signal
- **How strong** each frequency component is
- **The phase** (timing offset) of each component

```
Original Signal = Wave₁ (slow) + Wave₂ (medium) + Wave₃ (fast) + ...
```

### Visual Intuition

```
Time Domain:                    Frequency Domain:
     /\    /\                        |
    /  \  /  \                       |   |
   /    \/    \      ──FT──>         |   |   |
  /            \                     |   |   |
 ──────────────────              ────┴───┴───┴────
    (complex wave)               f1  f2  f3 (peaks at each freq)
```

**Think of it like this:** You hear a chord on a piano. The Fourier Transform is like having perfect pitch - it tells you exactly which notes (frequencies) make up that chord.

---

## The Mathematics

### Continuous Fourier Transform (CFT)

For a continuous signal $f(t)$:

$$
\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) \cdot e^{-i\omega t} \, dt
$$

Where:
- $\hat{f}(\omega)$ = Fourier transform at frequency $\omega$
- $e^{-i\omega t} = \cos(\omega t) - i\sin(\omega t)$ (Euler's formula)

**Inverse Fourier Transform (get back the original signal):**

$$
f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\omega) \cdot e^{i\omega t} \, d\omega
$$

### Why $e^{-i\omega t}$?

The complex exponential $e^{-i\omega t}$ is a "probe" for frequency $\omega$:

$$
e^{-i\omega t} = \cos(\omega t) - i\sin(\omega t)
$$

When you multiply your signal by this probe and integrate:
- If frequency $\omega$ exists in the signal → large value (constructive)
- If frequency $\omega$ doesn't exist → cancels out (destructive)

```mermaid
flowchart LR
    A[Signal f(t)] --> B[Multiply by e^-iωt]
    B --> C[Integrate over all t]
    C --> D{Large value?}
    D -->|Yes| E[Frequency ω is present]
    D -->|No| F[Frequency ω is absent]
```

---

## Discrete Fourier Transform (DFT)

For digital/sampled data with N points:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi kn / N}
$$

Where:
- $x[n]$ = input signal (N samples)
- $X[k]$ = output at frequency bin k
- $k = 0, 1, ..., N-1$

### What Each Frequency Bin Means

| Bin $k$ | Frequency | Interpretation |
|---------|-----------|----------------|
| 0 | 0 Hz (DC) | Average value of signal |
| 1 | $f_s/N$ | Lowest non-zero frequency |
| N/2 | $f_s/2$ | Nyquist frequency (highest) |

Where $f_s$ = sampling rate.

### DFT Output Interpretation

The DFT output $X[k]$ is a **complex number**:
- $|X[k]|$ = **Magnitude** (how strong this frequency is)
- $\angle X[k]$ = **Phase** (how shifted this frequency component is)

---

## Fast Fourier Transform (FFT)

The FFT is an **algorithm** (not a different transform) that computes the DFT efficiently.

| Method | Complexity | N=1024 operations |
|--------|------------|-------------------|
| Naive DFT | $O(N^2)$ | ~1,000,000 |
| FFT | $O(N \log N)$ | ~10,000 |

### How FFT Works (Cooley-Tukey)

The key insight: Split the DFT into smaller DFTs recursively.

```python
def fft_intuition(x):
    N = len(x)
    if N == 1:
        return x
    
    # Split into even and odd indices
    even = fft(x[0::2])  # x[0], x[2], x[4], ...
    odd = fft(x[1::2])   # x[1], x[3], x[5], ...
    
    # Combine using "twiddle factors"
    # W_k = e^(-2πik/N)
    return combine(even, odd)
```

**Divide and conquer:** Turn one N-point DFT into two N/2-point DFTs.

---

## Python Implementation

### Using NumPy

```python
import numpy as np

# Create a signal: 5 Hz + 12 Hz components
sample_rate = 100  # Hz
t = np.linspace(0, 1, sample_rate, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)

# Compute FFT
fft_result = np.fft.fft(signal)

# Get frequencies for each bin
frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)

# Get magnitude spectrum (only positive frequencies matter for real signals)
magnitude = np.abs(fft_result)

# Find peaks (should see spikes at 5 Hz and 12 Hz)
positive_freqs = frequencies[:len(frequencies)//2]
positive_magnitude = magnitude[:len(magnitude)//2]
```

### Visualizing the Spectrum

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Time domain
axes[0].plot(t, signal)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Time Domain Signal')

# Frequency domain
axes[1].stem(positive_freqs, positive_magnitude)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Frequency Domain (FFT)')

plt.tight_layout()
plt.show()
```

---

## 2D Fourier Transform (for Images)

Images are 2D signals. Apply FFT along both dimensions:

$$
F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-i2\pi(ux/M + vy/N)}
$$

### Image Frequency Interpretation

| Frequency | Corresponds To |
|-----------|----------------|
| Low frequencies | Smooth regions, gradual changes |
| High frequencies | Edges, fine details, noise |
| DC component (center) | Average brightness |

```python
import numpy as np
from PIL import Image

# Load grayscale image
img = np.array(Image.open('image.jpg').convert('L'))

# 2D FFT
f_transform = np.fft.fft2(img)

# Shift zero frequency to center (for visualization)
f_shifted = np.fft.fftshift(f_transform)

# Magnitude spectrum (log scale for visibility)
magnitude_spectrum = np.log(1 + np.abs(f_shifted))
```

---

## Applications in Machine Learning

### 1. Feature Engineering for Time Series

```python
def extract_frequency_features(signal, sample_rate):
    """Extract useful features from frequency domain."""
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    magnitudes = np.abs(fft_result[:len(signal)//2])
    
    features = {
        'dominant_freq': freqs[np.argmax(magnitudes)],
        'spectral_centroid': np.sum(freqs[:len(magnitudes)] * magnitudes) / np.sum(magnitudes),
        'spectral_bandwidth': np.sqrt(np.sum(((freqs[:len(magnitudes)] - features['spectral_centroid'])**2) * magnitudes) / np.sum(magnitudes)),
        'total_energy': np.sum(magnitudes**2),
    }
    return features
```

### 2. Audio Processing (Speech, Music)

- **Spectrograms**: Time-frequency representation (Short-Time Fourier Transform)
- **Mel-frequency cepstral coefficients (MFCCs)**: Common audio features
- **Speech recognition**: Models often work on frequency-domain inputs

```python
from scipy import signal

# Short-Time Fourier Transform (STFT) for spectrogram
frequencies, times, spectrogram = signal.stft(
    audio_signal, 
    fs=sample_rate, 
    nperseg=256,  # Window size
    noverlap=128  # Overlap between windows
)
```

### 3. Fourier Features for Positional Encoding

Transformers use Fourier-inspired positional encodings:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**Why?** Sinusoidal functions help the model learn relative positions.

```python
def positional_encoding(max_len, d_model):
    """Fourier-based positional encoding for transformers."""
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
```

### 4. Fourier Neural Operators (FNO)

Learn in frequency domain for solving PDEs and physical simulations:

```python
import torch
import torch.nn as nn

class FourierLayer(nn.Module):
    """Simplified Fourier layer for neural operators."""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes  # Number of Fourier modes to keep
        # Learnable weights in frequency domain
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))
    
    def forward(self, x):
        # x: (batch, channels, spatial_dim)
        # Transform to frequency domain
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant modes by learned weights
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :self.modes] = torch.einsum('bci,iom->bco', x_ft[..., :self.modes], self.weights)
        
        # Transform back
        return torch.fft.irfft(out_ft, n=x.size(-1))
```

### 5. Image Filtering and Preprocessing

```python
def low_pass_filter(image, cutoff_ratio=0.1):
    """Remove high-frequency noise from image."""
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular mask
    mask = np.zeros((rows, cols))
    r = int(min(rows, cols) * cutoff_ratio)
    y, x = np.ogrid[:rows, :cols]
    mask[(x - ccol)**2 + (y - crow)**2 <= r**2] = 1
    
    # Apply filter
    f_filtered = f_shifted * mask
    
    # Inverse transform
    f_ishift = np.fft.ifftshift(f_filtered)
    filtered_image = np.abs(np.fft.ifft2(f_ishift))
    return filtered_image
```

---

## Key Properties

### 1. Linearity

$$
\mathcal{F}\{a \cdot f(t) + b \cdot g(t)\} = a \cdot \mathcal{F}\{f(t)\} + b \cdot \mathcal{F}\{g(t)\}
$$

### 2. Time Shift → Phase Shift

$$
\mathcal{F}\{f(t - t_0)\} = e^{-i\omega t_0} \cdot \mathcal{F}\{f(t)\}
$$

Shifting in time only changes the phase, not magnitudes.

### 3. Convolution Theorem

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}
$$

**Convolution in time domain = Multiplication in frequency domain**

This is why FFT is used to speed up convolutions! Instead of $O(N^2)$ convolution, do:
1. FFT both signals: $O(N \log N)$
2. Multiply: $O(N)$
3. Inverse FFT: $O(N \log N)$

### 4. Parseval's Theorem (Energy Conservation)

$$
\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2
$$

Total energy in time domain = Total energy in frequency domain.

---

## Common Gotchas

### 1. Nyquist Frequency

You can only reliably detect frequencies up to **half your sampling rate**:

$$
f_{max} = \frac{f_s}{2}
$$

If your signal has higher frequencies, you get **aliasing** (false low frequencies appear).

### 2. Spectral Leakage

If your signal doesn't contain an exact integer number of cycles, you get "smearing" in the frequency domain.

**Solution:** Apply a window function (Hamming, Hanning) before FFT.

```python
window = np.hanning(len(signal))
windowed_signal = signal * window
fft_result = np.fft.fft(windowed_signal)
```

### 3. Zero Padding

Adding zeros to your signal before FFT gives finer frequency resolution (interpolation), but doesn't add new information.

```python
# Pad to next power of 2 (also faster for FFT)
padded_length = 2**int(np.ceil(np.log2(len(signal))))
padded_signal = np.pad(signal, (0, padded_length - len(signal)))
```

---

## Summary Table

| Concept | Time/Space Domain | Frequency Domain |
|---------|-------------------|------------------|
| Representation | $f(t)$ or $f(x, y)$ | $\hat{f}(\omega)$ or $F(u, v)$ |
| Shows | How signal changes over time | Which frequencies are present |
| Convolution | $O(N^2)$ operation | Simple multiplication |
| Filtering | Difficult | Easy (mask frequencies) |
| Pattern detection | Hard for periodic patterns | Peaks at periodic frequencies |

---

## When to Use Fourier Transform

| Use Case | Why FFT Helps |
|----------|---------------|
| Audio/speech analysis | Separate frequencies, find pitch |
| Image processing | Filter noise, detect edges, compression (JPEG) |
| Time series | Find periodicities, seasonal patterns |
| Signal denoising | Remove high-frequency noise |
| Data compression | Keep only important frequencies |
| Accelerating convolutions | Multiplication instead of sliding window |

---

## Resources

- [3Blue1Brown: But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) - Excellent visual intuition
- [Cooley-Tukey FFT Algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)
- [Stanford EE261: The Fourier Transform and its Applications](https://see.stanford.edu/Course/EE261)
