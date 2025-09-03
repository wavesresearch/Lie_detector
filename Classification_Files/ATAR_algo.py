import numpy as np
import pywt
from scipy.signal import butter, filtfilt

def bandpower(psd, freqs, band):
   
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])

def atar_algorithm(eeg_signal, fs=250, wavelet='db3', window_size=256, 
                   mode='elimination', alpha_bounds=[10, 100], 
                   beta=0.1, ipr=50, overlap=0.5):
    """
    Automatic and Tunable Algorithm for EEG Artifact Removal using Wavelet Decomposition
    
    Parameters:
    -----------
    eeg_signal : array_like
        Single channel EEG signal (1D array)
    fs : int
        Sampling frequency (Hz)
    wavelet : str
        Wavelet family to use (default: 'db3')
    window_size : int
        Size of processing window in samples (default: 256)
    mode : str
        Filtering mode: 'elimination', 'attenuation', or 'soft_thresholding'
    alpha_bounds : list
        Bounds for threshold parameter α [min, max]
    beta : float
        Threshold selection parameter (default: 0.1)
    ipr : float
        Interpercentile range for threshold calculation (default: 50)
    overlap : float
        Window overlap proportion (0-1) (default: 0.5)
    
    Returns:
    --------
    corrected_signal : ndarray
        Artifact-corrected EEG signal
    """
    
    # Step 1: High-pass filter the input signal (1Hz cutoff)
    corrected_signal = highpass_filter(eeg_signal, fs, cutoff=1.0)
    
    # Initialize parameters based on mode
    if mode == 'elimination':
        def filter_func(coeffs, alpha):
            return np.where(np.abs(coeffs) > alpha, 0, coeffs)
        gamma = 2 * np.mean(alpha_bounds)  # γ = 2α
    elif mode == 'attenuation':
        def filter_func(coeffs, alpha):
            return np.where(np.abs(coeffs) > alpha, coeffs * (alpha/np.abs(coeffs)), coeffs)
        gamma = 0.8 * np.mean(alpha_bounds)  # γ = 0.8α
    elif mode == 'soft_thresholding':
        def filter_func(coeffs, alpha):
            return np.sign(coeffs) * np.maximum(np.abs(coeffs) - alpha, 0)
        gamma = np.mean(alpha_bounds)  # γ = α
    else:
        raise ValueError("Invalid mode. Choose 'elimination', 'attenuation', or 'soft_thresholding'")
    
    # Process signal in overlapping windows
    step_size = int(window_size * (1 - overlap))
    n_samples = len(corrected_signal)
    output_signal = np.zeros_like(corrected_signal)
    window_weights = np.zeros_like(corrected_signal, dtype=float)
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        
        # Extract window (Step 3)
        window = corrected_signal[start:end]
        
        # Step 4: Compute L-level WPD
        wp = pywt.WaveletPacket(data=window, wavelet=wavelet, mode='symmetric')
        nodes = [node.path for node in wp.get_level(wp.maxlevel, 'freq')]
        
        # Get all coefficients
        coeffs = []
        for node in nodes:
            coeffs.append(wp[node].data)
        
        # Step 5: Compute threshold α (Eq. 16 in paper)
        # Here we implement a simplified version based on the description
        all_coeffs = np.concatenate(coeffs)
        alpha = compute_alpha_threshold(all_coeffs, alpha_bounds, beta, ipr)
        
        # Step 6: Apply wavelet filtering
        filtered_coeffs = []
        for c in coeffs:
            filtered_coeffs.append(filter_func(c, alpha))
        
        # Update the wavelet packet with filtered coefficients
        for i, node in enumerate(nodes):
            wp[node].data = filtered_coeffs[i]
        
        # Step 7: Reconstruct signal with IWPD
        reconstructed_window = wp.reconstruct(update=False)
        
        # Step 8: Overlap-add method for synthesis
        output_signal[start:end] += reconstructed_window
        window_weights[start:end] += 1
    
    # Normalize by window weights to account for overlapping
    window_weights[window_weights == 0] = 1  # avoid division by zero
    corrected_signal = output_signal / window_weights
    
    return corrected_signal

def highpass_filter(signal, fs, cutoff=1.0, order=4):
    """Butterworth highpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def compute_alpha_threshold(coeffs, alpha_bounds, beta, ipr):
    """
    Compute the threshold parameter α based on coefficient distribution
    
    Parameters:
    -----------
    coeffs : array_like
        Wavelet packet coefficients
    alpha_bounds : list
        Minimum and maximum bounds for α
    beta : float
        Scaling parameter for threshold calculation
    ipr : float
        Interpercentile range (percentage)
    
    Returns:
    --------
    alpha : float
        Calculated threshold value
    """
    # Calculate interpercentile range
    lower_percentile = (100 - ipr) / 2
    upper_percentile = 100 - lower_percentile
    lower_val = np.percentile(coeffs, lower_percentile)
    upper_val = np.percentile(coeffs, upper_percentile)
    
    # Simplified threshold calculation based on paper description
    alpha = beta * (upper_val - lower_val)
    
    # Constrain within bounds
    alpha = np.clip(alpha, alpha_bounds[0], alpha_bounds[1])
    
    return alpha

# Example usage:
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Ou 'TkAgg' si interface disponible
    import matplotlib.pyplot as plt
    
    # Generate a synthetic EEG signal with artifacts
    fs = 250  # sampling frequency
    t = np.arange(0, 5, 1/fs)  # 5 seconds of data
    clean_eeg = 50 * np.sin(2 * np.pi * 10 * t)  # 10 Hz neural activity
    artifacts = 200 * (np.random.rand(len(t)) < 0.01) * np.random.randn(len(t))  # sparse spikes
    noisy_eeg = clean_eeg + artifacts
    
    # Apply ATAR algorithm
    corrected_eeg = atar_algorithm(
        noisy_eeg, 
        fs=fs, 
        wavelet='db3', 
        window_size=256,
        mode='elimination', 
        alpha_bounds=[10, 100], 
        beta=0.1, 
        ipr=50
    )
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_eeg, label='Clean EEG')
    plt.title('Clean EEG Signal')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_eeg, label='Noisy EEG')
    plt.title('EEG with Artifacts')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, corrected_eeg, label='Corrected EEG')
    plt.title('After ATAR Algorithm')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('atar_algorithm_results.png')