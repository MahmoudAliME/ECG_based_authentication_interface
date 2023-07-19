import streamlit as st
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
import scipy.signal
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import statsmodels.api as sm
import scipy.fftpack as fft
import pywt

def AC_DCT(signal):   
	ac_dct_signal = np.zeros((len(signal),100))  # Array to store AC DCT coefficients
    # Iterate over each segment of the signal
	for i in range(len(signal)):
		segment = signal[i]

		# Iterate over each channel in the segment

		channel = segment[:100]

		    # Auto-Correlation
		ac = sm.tsa.acf(channel, nlags=1000)

		    # Discrete Cosine Transform
		dct = fft.dct(ac, type=2)

		    # Store AC DCT coefficients in the array
		ac_dct_signal[i, :100] = dct

  
	return ac_dct_signal


def dwt(signal):
 
  ecg_signal = signal 
  features = []
  for segment in range(len(signal)):
    # Perform Discrete Wavelet Transform (DWT)
    wavelet = 'db4'  # Wavelet type 
    level = 5  # Number of DWT decomposition levels

    coeffs = pywt.wavedec(signal[segment], wavelet, level=level)
    
    features.append(np.array(coeffs[0]))
  

  return features