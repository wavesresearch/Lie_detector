import mne
from mne import find_events, pick_events, Epochs
from mne.decoding import CSP
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Or 'TkAgg' 
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import logging
import pywt
import asrpy
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import ICA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.stats import ttest_ind
from sklearn.model_selection import GroupKFold, KFold
import tensorflow as tf
# from tensorflow.keras.models import Sequential  # Removed duplicate import
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, 
                                    Dense, Dropout, Flatten, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mne.time_frequency import tfr_morlet
import ATAR_algo as atar
import CSP_Algo as csp_algo
from scipy.stats import skew, kurtosis
from sklearn.impute import SimpleImputer
from scipy.signal import welch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SafeCSP(CSP):
    def transform(self, X):
        X_transformed = super().transform(X)
        X_transformed = np.where(X_transformed <= 1e-10, 1e-10, X_transformed)
        return np.log(X_transformed)
    
def map_event_ids(events):

    mapping = {50: 0, 51: 1, 2: 0, 3: 1}
    events[:, 2] = [mapping.get(code, code) for code in events[:, 2]]
    return events



def extract_raw_data(raw_fnames, event_id=dict(lie=0, true=1), tmin=0, tmax=1, baseline=(None, 0.5)):
        raws = []
        all_epochs = []

        for raw_fname in raw_fnames:
            raw = mne.io.read_raw_fif(raw_fname, preload=True)

            stim_channel = None
            for possible_name in ['stim', 'STI 014', 'STI 04']:
                if possible_name in raw.info['ch_names']:
                    stim_channel = possible_name
                    break

            event_id = {"true": 0, "lie": 1}

            events = mne.find_events(raw, stim_channel=stim_channel)
            events = mne.pick_events(events, include=[event_id['true'], event_id['lie']])

            ch_names=['F3','FC5','T7','C3','CP5','TP9', 'P5','P6', 'PO3', 'POz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'Pz', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2', 'CP6', 'TP10', 'PO7', 'PO8']
            channel_of_interest = ['CP5','CP6','T7','Pz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2','TP10','PO7','PO8'] 

            #events = mne.pick_events(events, include=[event_id['true'],event_id['lie']]) 
            epochs = mne.Epochs(raw, events, event_id = event_id, tmin=tmin, tmax=tmax, proj=True, baseline=baseline, reject=None, preload=True)

            #create global raws and epochs with all files (.fif)
            raws.append(raw)
            all_epochs.append(epochs)

        groups = []
        for idx, epochs in enumerate(all_epochs):
            groups.extend([idx] * len(epochs))

        groups = np.array(groups)
        # Final data without time truncation
        raw_combined = mne.concatenate_raws(raws)
        
        epochs_combined = mne.concatenate_epochs(all_epochs)
        print("Total number of epochs: ", len(epochs_combined))
        return raw_combined, epochs_combined, groups


def extract_raw_data_unified(fnames, tmin=0, tmax=1, baseline=(None, 0.5)):
    raws = []
    all_epochs = []
    groups = []

    for idx, fname in enumerate(fnames):
        raw = mne.io.read_raw_fif(fname, preload=True)

        # === Automatically find the stim channel ===
        stim_channel = None
        for cand in ["stim", "STI 014", "STI 04", "STIM"]:
            if cand in raw.ch_names:
                stim_channel = cand
                break
        if stim_channel is None:
            raise ValueError(f"No stimulation channel found in {fname}")

        events = mne.find_events(raw, stim_channel=stim_channel, verbose=False)

        unique_events = np.unique(events[:, 2])

        # === Mapping unified ===
        if set(unique_events) == {0, 1}:
            mapping = {0: 0, 1: 1}
        elif set(unique_events) == {2, 3}:
            mapping = {2: 0, 3: 1}
        elif set(unique_events) == {50, 51}:
            mapping = {50: 0, 51: 1}
        elif set(unique_events) == {49, 50, 51}:  
            mapping = {50: 0, 51: 1}  
        else:
            raise ValueError(f"Unexpected events in {fname}: {unique_events}")

        mask = np.isin(events[:, 2], list(mapping.keys()))
        events = events[mask]
        for k, v in mapping.items():
            events[events[:, 2] == k, 2] = v

        raw.pick_types(eeg=True)

        epochs = mne.Epochs(raw, events, event_id={'true': 0, 'lie': 1},
                            tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True)

        raws.append(raw)
        all_epochs.append(epochs)
        groups.extend([idx] * len(epochs))

   
    common_ch = set(all_epochs[0].info['ch_names'])
    for ep in all_epochs[1:]:
        common_ch &= set(ep.info['ch_names'])
    common_ch = list(common_ch)
    all_epochs = [ep.pick(common_ch) for ep in all_epochs]

    raw_combined = mne.concatenate_raws(raws)
    epochs_combined = mne.concatenate_epochs(all_epochs)
    groups = np.array(groups)

    return raw_combined, epochs_combined, groups




def classify_epochs(epochs_data, y, groups, n_splits=4):
    """
    epochs_data : ndarray (n_epochs, n_channels, n_times)
    y           : ndarray (n_epochs,)
    groups      : list or array (sessions/files per epoch)
    """
    X = epochs_data


    models = {
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # Verify the groups
    unique_groups = np.unique(groups)
    if len(unique_groups) > 1:
        print(f"Utilisation de GroupKFold avec {len(unique_groups)} groupes.")
        splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
        split_args = (X, y, groups)
    else:
        print("Un seul groupe détecté → utilisation de KFold standard.")
        splitter = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=42)
        split_args = (X, y)

    for name, model in models.items():
        print(f"\n=== {name} ===")

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Reshape
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        print("Confusion Matrix:")
        print(confusion_matrix(all_y_true, all_y_pred))
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred))
        print(f"Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")



def extract_statistical_features(data,fft=None,psd=None,dwt=None):

    n_epochs, n_channels, n_times = data.shape
    # 13 features per channel:
    # Preallocate feature array
    features = []
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)}   

    for epoch in data:
        # epoch shape: (n_channels, n_times)
        epoch_features = []
        for channel in epoch:
            segment = channel
            if fft == True:
                segment = np.abs(np.fft.rfft(channel, axis=0))
            # Compute features per channel
            ch_features = [
                    np.mean(segment),
                    np.median(segment),
                    np.std(segment),
                    np.var(segment),
                    skew(segment),
                    kurtosis(segment),
                    np.min(segment),
                    np.max(segment),
            ]

            freqs, psd = welch(segment, fs=250, nperseg=min(256, n_times))
            freq_feats = [atar.bandpower(psd, freqs, band) for band in bands.values()]

            epoch_features.extend(ch_features + freq_feats)
        features.append(epoch_features)
    
    return np.array(features)

def extract_fft_features(X, sfreq=128):
    """
    Compute FFT and extract statistical features for each channel.

    Args:
        X (ndarray): EEG data of shape (n_epochs, n_channels, n_times)
        sfreq (int): Sampling frequency in Hz

    Returns:
        features (ndarray): Shape (n_epochs, total_features)
    """
    n_epochs, n_channels, n_times = X.shape
    all_features = []

    for i in range(n_epochs):
        epoch_features = []

        for c in range(n_channels):
            signal = X[i, c, :]
            fft_values = np.abs(np.fft.fft(signal))[:n_times // 2]
            freqs = np.fft.fftfreq(n_times, d=1/sfreq)[:n_times // 2]

            epoch_features.extend([
                np.mean(fft_values),
                np.median(fft_values),
                np.std(fft_values),
                np.max(fft_values),
                skew(fft_values),
                kurtosis(fft_values)
            ])

        all_features.append(epoch_features)

    return np.array(all_features)

def extract_dwt_features(X, wavelet='db4', level=3):
    """
    Compute DWT and extract statistical features for each channel.

    Args:
        X (ndarray): EEG data of shape (n_epochs, n_channels, n_times)
        wavelet (str): Wavelet type
        level (int): Decomposition level

    Returns:
        features (ndarray): Shape (n_epochs, total_features)
    """
    n_epochs, n_channels, n_times = X.shape
    all_features = []

    for i in range(n_epochs):
        epoch_features = []

        for c in range(n_channels):
            signal = X[i, c, :]
            coeffs = pywt.wavedec(signal, wavelet, level=level)

            for coeff in coeffs:
                epoch_features.extend([
                    np.mean(coeff),
                    np.median(coeff),
                    np.std(coeff),
                    np.max(coeff),
                    skew(coeff),
                    kurtosis(coeff)
                ])

        all_features.append(epoch_features)

    return np.array(all_features)




def extract_epochs_multiple_channels(fnames, channel_names, event_id=dict(lie=50, true=51), tmin=0, tmax=1):
    import mne
    import numpy as np

    epochs_list = []
    labels = []

    for fname in fnames:
        raw = mne.io.read_raw_fif(fname, preload=True)

        
        raw.pick_channels(channel_names + ['STI 014'])

        
        events = mne.find_events(raw, stim_channel='STI 014')
        events = mne.pick_events(events, include=[event_id['lie'], event_id['true']])

        
        raw.drop_channels(['STI 014'])

        
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=(None, 0.5), preload=True)

        
        epochs_list.append(epochs.get_data())     # (n_epochs, n_channels, n_times)
        labels.append(epochs.events[:, -1])

    
    X = np.concatenate(epochs_list, axis=0)  # (N, C, T)
    y = np.concatenate(labels)               # (N,)

    return X, y


raw_fnames_P001 = [ 
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-001_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-002_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-003_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-004_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-001_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-002_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-003_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-004_eeg.fif"
    ]

raw_fnames_P003 = [    
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-001_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-002_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-003_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-004_eeg.fif"
              ]


raw_fnames = [ 
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-001_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-002_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-003_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P001_ses-S001_task-Default_run-004_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P002_ses-S001_task-Default_run-001_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P002_ses-S001_task-Default_run-002_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-001_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-002_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-003_eeg.fif",
    "/home/dutailly/Lie_detector/Dataset/sub-P003_ses-S001_task-Default_run-004_eeg.fif",
    
              ]


fname_session = ["/home/dutailly/Code_Perso_Lie_Detector/NTNU_Dataset/sub-P004_ses-S001_task-Default_run-001_eeg.fif",
                 "/home/dutailly/Code_Perso_Lie_Detector/NTNU_Dataset/sub-P005_ses-S001_task-Default_run-001_eeg.fif",
                 "/home/dutailly/Code_Perso_Lie_Detector/NTNU_Dataset/sub-P006_ses-S001_task-Default_run-001_eeg.fif",
                  
                                           ]


channel_of_interest = ['CP5','CP6','T7','Pz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2','TP10','PO7','PO8'] 
all_channel = ['F3','FC5','T7','C3','CP5','TP9', 'P5','P6', 'PO3', 'POz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'Pz', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2', 'CP6', 'TP10', 'PO7', 'PO8']
p_test_channel = ['T8', 'TP10', 'AFz', 'Fz', 'T7']
lie_wave_channel = ['F3', 'T7', 'POz', 'T8','F4']


# Raw signal + pick_channels + filter + ATAR
raw_combined, epochs_combined, groups = extract_raw_data_unified(raw_fnames, tmin=0, tmax=1, baseline=(None, 0.5))
epochs_combined.pick_channels(channel_of_interest)  # Pick only channels of interest
epochs_filtered = epochs_combined.copy().filter(l_freq=0.5, h_freq=45.0, method='iir')
epochs_atar = atar.atar_algorithm(epochs_filtered, wavelet='db4', mode='attenuation', alpha_bounds=(0.1, 0.5), beta=0.5, ipr=0.1)
print("Shape of epochs after ATAR:", epochs_atar.shape)
y_base = epochs_combined.events[:, -1]
le = LabelEncoder()
y = le.fit_transform(y_base)
#print("Encoded labels:", y,len(y))
X_atar = epochs_atar.reshape(len(epochs_atar), -1)  # Reshape epochs to 2D array for ATAR
print("Shape of X_atar:", X_atar.shape)


#X_dwt = extract_dwt_features(epochs_atar, wavelet='db4', level=3)
#print("Shape of X_dwt:", X_dwt.shape)


#X_features = extract_statistical_features(epochs_atar, fft=None, psd=None, dwt=None)
#print("Shape of X_features:", X_features.shape)


csp_mne = mne.decoding.CSP(n_components=4, reg=None, log=False, norm_trace=False)
csp_mne.fit(epochs_atar, y)
X_csp_mne = csp_mne.transform(epochs_atar)
print("Shape of X_csp_mne:", X_csp_mne.shape)


csp = csp_algo.CSP_L1()
csp.fit(epochs_atar, y)
X_csp = csp.transform(epochs_atar)
X_csp = np.nan_to_num(X_csp, nan=0.0, posinf=0.0, neginf=0.0)
print("Shape of X_csp:", X_csp.shape)


pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_atar)
print("Shape of X_pca:", X_pca.shape)


classify_epochs(X_csp_mne,y, groups)