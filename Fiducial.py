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

def pan_tompkins(ecg, sampling_rate):
  
  # Bandpass filter the ECG signal.
  w1 = 5 * 2 / sampling_rate
  w2 = 15 * 2 / sampling_rate
  b, a = scipy.signal.butter(4, [w1, w2], 'bandpass')
  ecg_bp = scipy.signal.filtfilt(b, a, ecg)

  # Compute the first derivative of the filtered ECG signal.
  d_ecg_bp = scipy.fftpack.diff(ecg_bp)

  # Compute the squared magnitude of the first derivative.
  m = np.square(d_ecg_bp)

  # Compute the threshold.
  threshold = np.mean(m) + 3 * np.std(m)

  # Find the peaks in the squared magnitude.
  peaks = np.where(m > threshold)[0]
  #print(peaks)
  # Return the QRS locations.
  return peaks


def find_local_maximum(ecg_signal, qrs_onset_index, sampling_rate, search_window_width=0.2):
    search_window_samples = int(search_window_width * sampling_rate)
    search_window_start = qrs_onset_index
    search_window_end = qrs_onset_index - search_window_samples

    # Adjust the search window if it exceeds the signal boundaries
    if search_window_end < 0:
        search_window_end = 0

    search_window = ecg_signal[search_window_end:search_window_start]
    local_max_index = np.argmax(search_window)

    # Convert the local maximum index back to the original signal index
    local_max_index = search_window_end + local_max_index

    return local_max_index



def detect_t_wave(ecg_signal, offset_index, sampling_rate, search_window_width=0.4):
    search_window_samples = int(search_window_width * sampling_rate)
    search_window_start = offset_index
    search_window_end = offset_index + search_window_samples

    search_window = ecg_signal[search_window_start:search_window_end]


    # Finding the local maximum within the search window
    t_wave_index = np.argmax(search_window)

    # Calculate the index within the original signal
    t_wave_index += search_window_start

    return t_wave_index



def detect_wave_onset_offset(ecg_signal, p_peak_index, sampling_rate, search_window_width=0.2, offset_window_width=0.05):
    search_window_samples = int(search_window_width * sampling_rate)
    search_window_start = p_peak_index - search_window_samples
    search_window_end = p_peak_index + search_window_samples

    search_window = ecg_signal[search_window_start:search_window_end]

    # Find the onset point
    if len(search_window) != 0:
      p_wave_onset_index = np.argmin(search_window[:p_peak_index - search_window_start])
      p_wave_onset_index += search_window_start
    else:
      p_wave_onset_index = p_peak_index - 50

    # Create a narrower window for offset detection
    offset_window_samples = int(offset_window_width * sampling_rate)
    offset_search_window_start = p_peak_index
    offset_search_window_end = p_peak_index + offset_window_samples

    offset_search_window = search_window[offset_search_window_start - search_window_start:offset_search_window_end - search_window_start]
    if len(offset_search_window) != 0:
      # Find the offset point
      min_value = np.min(offset_search_window)
      offset_indices = np.where(offset_search_window == min_value)[0]
      p_wave_offset_index = offset_indices[0] + offset_search_window_start
    else:
      p_wave_offset_index = p_peak_index + 50
    return p_wave_onset_index, p_wave_offset_index


def extract_fiducial(ecg_segment, sampling_rate):
  
  # finding QRS Complex
  qrs_Points = pan_tompkins(ecg_segment, sampling_rate)
  qrs_onset, qrs_offset = qrs_Points[1], qrs_Points[-2]
  #print(qrs_onset, qrs_offset)

  # Finding P Wave (Peak, Onset, Offset)
  p_peak =  find_local_maximum(ecg_segment, qrs_onset, sampling_rate, search_window_width=0.2)
  #print("P")
  p_onset, p_offset = detect_wave_onset_offset(ecg_segment, p_peak, sampling_rate, search_window_width=0.2, offset_window_width=0.05)

  #print("T")
  # Finding T wave (Peak, Onset, Offset)
  t_peak = detect_t_wave(ecg_segment, qrs_offset, sampling_rate, search_window_width=0.4)
  t_onset, t_offset = detect_wave_onset_offset(ecg_segment, t_peak, sampling_rate, search_window_width=0.15, offset_window_width=0.08)

  return [p_peak, p_onset, p_offset, t_peak, t_onset, t_offset, qrs_onset, qrs_offset]

def transform_data(signal, sampling_rate):
  transformed_signal = []
  for segment in range(len(signal)):
      feature_vector = []
      feature_vector.append(extract_fiducial(signal[segment], sampling_rate))
      transformed_signal.append(feature_vector)
  return np.array(transformed_signal)