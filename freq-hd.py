# -*- coding: utf-8 -*-
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange, sin, pi, cos
import cmath
from clipselection import *

# Ideal low-pass filter
def low_pass_filter(img, radius):
    rows, cols = img.shape
    center = int(rows/2), int(cols/2)

    mask_low = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x-center[0])**2+(y-center[1])**2 <= radius*radius
    mask_low[mask_area] = 1
    mask_high = 1 - mask_low
    return mask_low, mask_high

# Ideal high-pass filter
def gaus_filter(img, radius):
    rows, cols = img.shape
    center = int(rows/2), int(cols/2)

    mask_low = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = (i-center[0])**2+(j-center[1])**2
            mask_low[i, j] = np.exp(-0.5*dist/(radius**2))
    mask_high = 1 - mask_low
    return mask_low, mask_high

# Butterworth low-pass filter
def bw_filter(img, radius, n=2):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask_low = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask_low[i, j] = 1/(1+(dist/radius)**(n/2))
    mask_high = 1 - mask_low
    return mask_low, mask_high


# Notch filter
def notch_filter(img, h, w):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            mask[u,v]=0
    for u in range(rows):
        for v in range(cols):
            if abs(u - center[0]) < h and abs(v - center[1]) < w:
                mask[u, v] = 1

    return mask

# Spatial discrete fourier transform
def spatial_dft(color_img):
    gray_img=cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

    # Performing the DFT on the spatial domain
    # performing DFT
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # shift the low frequency components to the center
    dft_shift = np.fft.fftshift(dft)
    # calculate the magnitude
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Filtering
    # setting the threshold
    threshold_freq_spatial = 20
    # low-pass and high-pass
    mask_low, mask_high = gaus_filter(magnitude_spectrum, threshold_freq_spatial)
    low_freq = dft_shift * mask_low
    high_freq = dft_shift * mask_high
    low_freq_abs = cv2.magnitude(low_freq[:,:,0], low_freq[:,:,1])
    high_freq_abs = cv2.magnitude(high_freq[:,:,0], high_freq[:,:,1])

    # Performing the IDFT
    ifft_shift2_low = np.fft.ifftshift(low_freq)  
    result_low = cv2.idft(ifft_shift2_low)
    ifft_shift2_high = np.fft.ifftshift(high_freq)  
    result_high = cv2.idft(ifft_shift2_high)

    return magnitude_spectrum, low_freq_abs, high_freq_abs, result_low[:,:,0], result_high[:,:,0]

# Temporal discrete fourier transform
def temporal_dft(img_clip): 
    
    N = img_clip.shape[0]
    H = img_clip.shape[1]
    W = img_clip.shape[2]
    K = N

    freq_amplitude = 0
    for h in range(H):
        for w in range(W):
            # Calculate the Temporal DFT at the spatial location (h, w)
            Xk = []
            for k in range(K):
                tmp_k = 0
                for n in range(N):
                    cm = -1j
                    cm *= 2 * pi * k * n / N
                    tmp_k += img_clip[n, h, w] * cmath.exp(cm)
                Xk.append(abs(tmp_k) ** 2)
            threshold_freq_temporal = K // 2
            freq_amplitude += np.sum(Xk[threshold_freq_temporal:K])

    freq_amplitude /= H * W * N

    return freq_amplitude

# Load data
base_path = r'pic/Happy0010'
files = os.listdir(base_path)
if '.DS_Store' in files:
    files.remove('.DS_Store')
files.sort(key=lambda x: int(x.split('.')[0]))

# Establish arrays to store the frequency domain and spatial domain spectra after filtering
all_freq_low_s = np.empty([1,224,224])
all_freq_high_s = np.empty([1,224,224])
all_spat_low = np.empty([1,224,224])
all_spat_high = np.empty([1,224,224])


# Spatial DFT
for path in files:  # Iterate through the images in the sequence and perform spatial DFT
    # Obtain the original spatial domain image
    full_path = os.path.join(base_path, path)
    color_img = cv2.imread(full_path)

    # spatial domain <-> frequency domain
    freq, freq_low_s, freq_high_s, spat_low, spat_high = spatial_dft(color_img) # Obtain the frequency domain and spatial domain spectra after filtering

    # store the frequency and spatial domain spectra after filtering, noting that the first element is a matrix entirely of zeros
    all_freq_low_s = np.append(all_freq_low_s, [freq_low_s], axis=0)    # (N, H, W, 2)
    all_freq_high_s = np.append(all_freq_high_s, [freq_high_s], axis=0) # (N, H, W, 2)
    all_spat_low = np.append(all_spat_low, [spat_low], axis=0)  # (N, H, W)
    all_spat_high = np.append(all_spat_high, [spat_high], axis=0)   # (N, H, W)
    
# Temporal DFT
step = 4
A_low = []
A_high = []
for i in range(1, all_spat_low.shape[0] - step + 1):
    A_low.append(temporal_dft(all_spat_low[i:i+step, :, :])) 
    A_high.append(temporal_dft(all_spat_high[i:i+step, :, :]))

# Draw the STFA module result diagram
x = range(len(A_low)) 
plt.plot(x, A_low, color='orangered', marker='o', linestyle='-', label='low frequency')
plt.plot(x, A_high, color='BlueViolet', marker='D', linestyle='-.', label='high frequency')
plt.legend()
plt.show()
plt.scatter(A_high, A_low, color= 'DimGray',marker='o')
plt.xlabel("high frequency")
plt.ylabel("low frequency")
plt.show()

# MBC module
softmaxres = clip_select(A_high, A_low)

clipweight = (np.array(A_high)).reshape(-1,1) / softmaxres

# Draw the final result diagram
x = range(len(clipweight)) 
plt.plot(x, clipweight, color='DarkBlue', marker='D', linestyle='-.', label='dynamicity weight')
plt.legend() 
plt.xlabel("X") 
plt.ylabel("Y") 
plt.show()
