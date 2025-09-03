# Import necessary libraries
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Or 'TkAgg' if GUI is available
import matplotlib.pyplot as plt
import os
import pandas as pd
import pywt
import tensorflow as tf
import logging
import asrpy
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import ICA
from sklearn.svm import SVC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
import seaborn as sns
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, 
                                    Dense, Dropout, Flatten, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from mne.time_frequency import tfr_morlet
import ATAR_algo as atar
from scipy.stats import skew, kurtosis, entropy
from sklearn.impute import SimpleImputer
from scipy.signal import welch


def load_all_csvs_with_labels(folder_path, label_txt_path):
    """
    Load all CSV files, reshape them, and add their labels.

    Args:
        folder_path (str): Folder containing the CSV files.
        label_txt_path (str): Path to the label text file.

    Returns:
        X (np.ndarray): EEG data of shape (n_files, 25, 384, 5)
        y (np.ndarray): Labels array of shape (n_files,)
    """
    # Load labels from text file
    label_df = pd.read_csv(label_txt_path, sep='\t')  # Be careful with tab separator
    label_df['FILE_NAME'] = label_df['SUBJECT'] + label_df['SESSION'] + ".csv"
    label_dict = dict(zip(label_df['FILE_NAME'], label_df['LIE/TRUTH']))

    # Read and process CSV files
    X_list = []
    y_list = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv") and file in label_dict:
            try:
                df = pd.read_csv(os.path.join(folder_path, file))  # shape (9600, 5)
                data = df.values.reshape(25, 384, 5)               # shape (25, 384, 5)
                X_list.append(data)
                y_list.append(label_dict[file])
                print(f"{file} loaded and transformed.")
            except Exception as e:
                print(f"Error with {file}: {e}")

    X = np.array(X_list)  # (n_files, 25, 384, 5)
    y = np.array(y_list)  # (n_files,)
    return X, y


def extract_fft_features(X):
    """
    Apply FFT and extract statistical features for each segment and channel.

    Args:
        X (ndarray): EEG data of shape (N, 25, 384, 5)

    Returns:
        features (ndarray): Shape (N*25, total_features)
    """
    N, n_segments, n_points, n_channels = X.shape
    all_features = []

    for i in range(N):
        for j in range(n_segments):
            segment = X[i, j, :, :]  # shape (384, 5)
            fft = np.abs(np.fft.rfft(segment, axis=0))  # shape (freq_bins, 5)
            features = []

            for c in range(n_channels):
                spectrum = fft[:, c]
                features.extend([
                    np.mean(spectrum),
                    np.median(spectrum),
                    np.std(spectrum),
                    np.max(spectrum),
                    skew(spectrum),
                    kurtosis(spectrum)
                ])
            all_features.append(features)

    return np.array(all_features)


def extract_psd_features(X, sfreq=128):
    """
    Compute PSD using Welch's method and extract statistical features for each EEG band and channel.

    Args:
        X (ndarray): EEG data of shape (N, 25, 384, 5)
        sfreq (int): Sampling frequency in Hz

    Returns:
        features (ndarray): Shape (N*25, total_features)
    """
    eeg_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    N, n_segments, n_points, n_channels = X.shape
    all_features = []

    for i in range(N):
        for j in range(n_segments):
            segment = X[i, j, :, :]  # (384, 5)
            segment_features = []

            for c in range(n_channels):
                signal = segment[:, c]
                freqs, psd = welch(signal, fs=sfreq, nperseg=128)

                for band, (fmin, fmax) in eeg_bands.items():
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    band_power = psd[idx_band]

                    segment_features.extend([
                        np.mean(band_power),
                        np.median(band_power),
                        np.std(band_power),
                        np.max(band_power),
                        skew(band_power),
                        kurtosis(band_power)
                    ])

            all_features.append(segment_features)

    return np.array(all_features)


def extract_dwt_features(X, wavelet='db4', level=4):
    """
    Extract DWT features for each segment and channel.

    Args:
        X (ndarray): (n_samples, n_segments, n_points, n_channels)
        wavelet (str): Wavelet type (e.g. 'db4')
        level (int): Decomposition level

    Returns:
        features (ndarray): (n_samples * n_segments, n_features)
    """
    n_samples, n_segments, n_points, n_channels = X.shape
    feature_list = []

    for i in range(n_samples):
        for j in range(n_segments):
            segment_features = []
            for ch in range(n_channels):
                signal = X[i, j, :, ch]
                coeffs = pywt.wavedec(signal, wavelet, level=level)
                for c in coeffs:
                    mean_ = np.mean(c)
                    std_ = np.std(c)
                    energy = np.sum(np.square(c))
                    ent = entropy(np.abs(c / np.sum(np.abs(c)) + 1e-12))
                    segment_features.extend([mean_, std_, energy, ent])
            feature_list.append(segment_features)

    return np.array(feature_list)

def extract_raw_reshape(X):
    """
    Reshape raw EEG data to a 2D array.

    Args:
        X (ndarray): EEG data of shape (n_samples, n_segments, n_points, n_channels)

    Returns:
        reshaped_X (ndarray): Reshaped data of shape (n_samples * n_segments, n_points * n_channels)
    """
    n_samples, n_segments, n_points, n_channels = X.shape
    reshaped_X = X.reshape(n_samples * n_segments, n_points * n_channels)
    return reshaped_X

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model for EEG classification.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_lstm_model(input_shape, num_classes):
    """
    Build a simple LSTM model for EEG classification.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm_model(input_shape):
    """
    Build a CNN-LSTM model for EEG classification.

    Args:
        input_shape (tuple): Input data shape (segments, time_points, channels)

    Returns:
        model: Compiled Keras model
    """
    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def build_cnn_lstm_model2(input_shape):
    """
    Alternative CNN-LSTM model using 2D convolution.
    """
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



def evaluate_classifiers(X, y, test_size=0.2, random_state=42):
    """
    Apply several classifiers and display the results.

    Parameters:
        X (ndarray): features (n_samples, n_features)
        y (ndarray): labels (n_samples,)
        test_size (float): proportion for the test set
        random_state (int): seed for reproducibility
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    classifiers = {
        "SVM": SVC(kernel='rbf', C=1, gamma='scale'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=random_state)
    }

    for name, clf in classifiers.items():
        print(f"\n=== {name} ===")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("Confusion Matrix :\n",confusion_matrix(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrix_{name}.png")
        plt.close(fig)


def evaluate_CNN_LSTM(X, y):
    """
    Train and evaluate CNN, LSTM, and CNN-LSTM models.

    Parameters:
        X (ndarray): input data shaped for CNN and LSTM
        y (ndarray): labels
    """
    # Data preparation
    X_cnn = X.reshape(-1, 384, 5, 1)  # For simple CNN
    X_lstm = X.reshape(-1, 384, 5)    # For simple LSTM
    X_cnn_lstm = X  # Shape: (n_samples, 25, 384, 5)

    # Expanding labels for simple CNN and LSTM (window by window)
    y_expanded = np.repeat(y, X.shape[1])  # (n_samples * 25,)

    # Label encoding
    y_encoded = LabelEncoder().fit_transform(y_expanded)
    y_cat = to_categorical(y_encoded)

    # Proper dataset splits
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cat, test_size=0.2, random_state=42)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_cat, test_size=0.2, random_state=42)

    # CNN Model
    cnn_model = build_cnn_model((384, 5, 1), num_classes=2)
    cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_split=0.2)

    y_pred_cnn = cnn_model.predict(X_test_cnn)
    y_pred_labels_cnn = np.argmax(y_pred_cnn, axis=1)
    y_true_labels_cnn = np.argmax(y_test_cnn, axis=1)

    print("Confusion Matrix CNN:")
    print(confusion_matrix(y_true_labels_cnn, y_pred_labels_cnn))
    print("\nClassification Report CNN:")
    print(classification_report(y_true_labels_cnn, y_pred_labels_cnn))

    # LSTM Model
    lstm_model = build_lstm_model((384, 5), num_classes=2)
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.2)

    y_pred_lstm = lstm_model.predict(X_test_lstm)
    y_pred_labels_lstm = np.argmax(y_pred_lstm, axis=1)
    y_true_labels_lstm = np.argmax(y_test_lstm, axis=1)

    print("Confusion Matrix LSTM:")
    print(confusion_matrix(y_true_labels_lstm, y_pred_labels_lstm))
    print("\nClassification Report LSTM:")
    print(classification_report(y_true_labels_lstm, y_pred_labels_lstm))

    # CNN-LSTM (on full samples, not individual windows)
    # No one-hot encoding here
    y_encoded_seq = LabelEncoder().fit_transform(y)  # Labels 0 or 1

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_cnn_lstm, y_encoded_seq, test_size=0.2, random_state=42)

    cnn_lstm_model = build_cnn_lstm_model((25, 384, 5))
    cnn_lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=8, validation_split=0.2)

    y_pred_seq = cnn_lstm_model.predict(X_test_seq)
    y_pred_labels_seq = (y_pred_seq > 0.5).astype(int).flatten()

    print("Confusion Matrix CNN-LSTM:")
    print(confusion_matrix(y_test_seq, y_pred_labels_seq))
    print("\nClassification Report CNN-LSTM:")
    print(classification_report(y_test_seq, y_pred_labels_seq))



# ------------------------# Main execution
    
folder_path = "/home/dutailly/Code_Perso_Lie_Detector/LieWaves_Dataset/Preprocessing/4_ATAR"
label_txt_path = "/home/dutailly/Code_Perso_Lie_Detector/SUBJECT_SESSION_LIETRUTH.txt"

X, y = load_all_csvs_with_labels(folder_path, label_txt_path)
y_extended = np.repeat(y, 25)
X_features = extract_dwt_features(X) 
print(f"Shape of X: {X.shape}")  # Should be (n_samples, 25, 384, 5)
print(f"Shape of X_features: {X_features.shape}")  # Should be (n_samples * 25, 384 * 5)

#evaluate_classifiers(X_features, y_extended)
#evaluate_CNN_LSTM(X, y)

