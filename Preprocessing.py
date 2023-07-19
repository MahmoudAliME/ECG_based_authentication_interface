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
from Fiducial import *
def filter_sig(record_name):
    # Load the ECG recording and associated metadata
    record = wfdb.rdrecord(record_name)

    # Extract the ECG signal from all channels
    ecg_signal = record.p_signal

    # Define the filter cut-off frequencies in Hz
    lowcut = 1.0
    highcut = 40.0

    # Define the filter order
    order = 4

    # Compute the Nyquist frequency
    nyq = 0.5 * record.fs

    # Compute the filter cut-off frequencies in normalized units
    low = lowcut / nyq
    high = highcut / nyq

    # Compute the filter coefficients using a Butterworth filter for each channel
    filtered_signal = np.zeros_like(ecg_signal)
    for i in range(ecg_signal.shape[1]):
        b, a = butter(order, [low, high], btype='band')
        filtered_signal[:, i] = filtfilt(b, a, ecg_signal[:, i])
    return filtered_signal

def detect_cutt_off(signal):
	p_peak, p_onset, p_offset, t_peak, t_onset, t_offset, qrs_onset, qrs_offset = extract_fiducial(signal[:1000], 1000)
	cut_off = 0
	if t_offset <= 950 :
		cut_off += t_offset
		p_peak, p_onset, p_offset, t_peak, t_onset, t_offset, qrs_onset, qrs_offset = extract_fiducial(signal[t_offset:(t_offset+600)], 1000)
    #visualize_fiducial(signal[cut_off:cut_off+600], p_peak, p_onset, p_offset, t_peak, t_onset, t_offset, qrs_onset, qrs_offset)
		return p_onset + cut_off
	elif p_onset >= 50:
		return p_onset
	else:
		return 150

def calculate_rr_intervals(signal):
  
  peaks = pan_tompkins(signal, 1000)

  iter = peaks[1]
  onset_offset = [iter,]
  for peak in range(2,len(peaks)-1):
    if peaks[peak] - iter > 5:
      iter = peaks[peak]
      onset_offset.append(iter)
    else:
      iter = peaks[peak]
  
  rr_intervals = []
  iter = [onset_offset[0], onset_offset[1]]
  for point in range(2,len(onset_offset)):
    if point % 2 == 0:
      rr_intervals.append(onset_offset[point] - iter[0])
      iter[0] = onset_offset[point]
    else:
      #rr_intervals.append(onset_offset[point] - iter[1])
      iter[1] = onset_offset[point]


  return rr_intervals
  
  
def segment_sig(signal, rr_intervals):
    intervals = rr_intervals

    # Create an empty numpy array to store the segments
    segments = []

    # Split the signal into segments
    start = 0
    for i in range(len(intervals)):
        end_idx = start + intervals[i]
        segments.append(signal[start:end_idx])
        start += intervals[i]
    return segments

def prepare_data(ac_dct_signal):
    shape = ac_dct_signal.shape[1:]
    dims = np.prod(shape)
    sig = np.zeros((len(ac_dct_signal),(dims)))
    for i in range(len(ac_dct_signal)): 
        sig[i] = ac_dct_signal[i].flatten()
  
    return sig

def PCA_sig(X, model):
	
    num_components = 8
    pca_signal = np.zeros((len(X),num_components))

    pca = joblib.load(model + ".h5")
    X= pca.transform(X)

    return X
    
    
def add_to_df(X_train,X_test,y_train,y_test,df):
    # Create dataframes for each array
    train_df = pd.DataFrame(X_train, columns=['Feature_{}'.format(i) for i in range(X_train.shape[1])])
    train_df['Target'] = y_train

    test_df = pd.DataFrame(X_test, columns=['Feature_{}'.format(i) for i in range(X_test.shape[1])])
    test_df['Target'] = y_test

    # Concatenate the train and test dataframes
    combined_df = pd.concat([df,train_df, test_df], axis=0).reset_index(drop=True)

    # Print the combined dataframe
    return combined_df

def prepare_data_dwt(ac_dct_signal):
 
    min_shape = 20
    reshaped_list = []
    for element in ac_dct_signal:
      # Truncate or pad the element to match the minimum shape
      reshaped_element = element[:min_shape] if len(element) >= min_shape else element + [0] * (min_shape - len(element))
      reshaped_list.append(reshaped_element)

    reshaped_list = np.array(reshaped_list)
    shape = reshaped_list.shape[1:]

    dims = np.prod(shape)
    sig = np.zeros((len(reshaped_list),(dims)))
    for i in range(len(reshaped_list)): 
        x= reshaped_list[i].flatten()
        sig[i] = x
    sig.shape
    X=sig[:,:dims]
    return X