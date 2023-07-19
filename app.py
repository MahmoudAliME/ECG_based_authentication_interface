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
from Non_Fiducial import *
from Preprocessing import *


    

  
def main():
	st.title("ECG Person Classification")
	
	
	# File upload section
	uploaded_file = st.file_uploader("Upload a .hea file", type=".hea")
	classification_type = st.radio("Choose classification type:", ("Fiducial", "AC-DCT", "DWT"))
	
	if uploaded_file is not None:
		# Read the uploaded file
		record_name = uploaded_file.name.split('.')[0]  # Extract the record name without the extension
		record = wfdb.rdrecord(record_name)  # Read the record using wfdb

		# Extract the signal from the record
		filtered_signal = filter_sig(record_name)

		cut_off = detect_cutt_off(filtered_signal[:, 13]) - 150

		rr_intervlas = calculate_rr_intervals(filtered_signal[cut_off:,13])

		segmented_signal = segment_sig(filtered_signal[cut_off:, 13], rr_intervlas)

		non_fiducial_signal = AC_DCT(segmented_signal)

		fiducial_signal = transform_data(segmented_signal, 1000)
		
		dwt_signal = dwt(segmented_signal)
		
		
		
		X_non_fiducial = prepare_data(non_fiducial_signal)
		X_fiducial = prepare_data(fiducial_signal)
		X_dwt = prepare_data_dwt(dwt_signal)

		X_non_fiducial = PCA_sig(X_non_fiducial, "models/pca_nonFiducial")
		X_dwt = PCA_sig(X_dwt, "models/pca_dwt")
		X_fiducial = PCA_sig(X_fiducial, "models/pca")
		
		
		model = joblib.load("models/SVM_DWT.h5")
		dwt_predictions = []
		for segment in range (len(X_dwt)):
			dwt_predictions.append(model.predict([X_dwt[segment]]))
		series3 = pd.Series(dwt_predictions)	
		dwt_result = series3.value_counts().max() / len(series3)
		
			
			
		fiducial_threshold = 0.9
		non_fiducial_threshold = 0.5
		model = joblib.load("models/non_fiducial_model.h5")
		Non_fiducial_predictions = []
		for segment in range (len(X_non_fiducial)):
			Non_fiducial_predictions.append(model.predict([X_non_fiducial[segment]]))
		
		series1 = pd.Series(Non_fiducial_predictions)

		# Get the value counts
		non_fiducial_result = series1.value_counts().max() / len(series1)


		
		# Perform the required processing steps
		model = joblib.load("models/SVM.h5")
		fiducial_predictions = []
		for segment in range (len(X_fiducial)):
			fiducial_predictions.append(model.predict([X_fiducial[segment]]))

		
		series2 = pd.Series(fiducial_predictions)
			
		# Get the value counts
		fiducial_result = series2.value_counts().max() / len(series2)



		st.subheader("Classification Result:")
		if classification_type == "Fiducial":
		# Perform Classification 1
			if fiducial_result >= fiducial_threshold:
				idx = int(series2.value_counts().idxmax()[0])
				st.markdown("**Classification Type:** Fiducial")
				st.success(f"**Recognized by the system using fiducial features .. User: {idx}**")
			
			else:
				st.error("This user is not recognized by the system!")
		elif classification_type == "AC-DCT":
		# Perform Classification 2
			if non_fiducial_result >= non_fiducial_threshold:
				idx = int(series1.value_counts().idxmax()[0])
				st.markdown("**Classification Type:** AC-DCT")
				st.success(f"**Recognized by the system using Non Fiducial features .. User:  {idx}**")
			else:
				st.error("This user is not recognized by the system!")
		elif classification_type == "DWT":
		# Perform Classification 2
			if dwt_result >= non_fiducial_threshold:
				idx = int(series3.value_counts().idxmax()[0])
				st.markdown("**Classification Type:** DWT")
				st.success(f"**Recognized by the system using DWT features .. User:  {idx}**")
			else:
				st.error("This user is not recognized by the system!")
            	
            	
            	
            	
			



if __name__ == '__main__':
    patients = ['patient174', 'patient180', 'patient185', 'patient214']
    records = ['s0325lre', 's0476_re', 's0336lre', 's0436_re']
    main()
