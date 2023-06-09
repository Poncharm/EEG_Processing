import mne
import numpy as np
from scipy import integrate

def load_and_filter_raw_data(file_path, l_freq=1.0, h_freq=50.0):
    """Load raw data from the specified file and apply bandpass filter."""
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw

def change_data_range(channel_data, start_index, end_index):
    """Filter the desired range of data and convert to microvolts."""
    return channel_data[start_index:end_index] * (10 ** 6)

def create_wavelet(window, a, b, f_max):
    """Create a Morlet wavelet based on the provided parameters."""
    wavelet = (1 / a ** 0.5) * np.exp(-((((window - b) / a) ** 2) / 2)) * (np.cos(2 * np.pi * f_max * (window - b) / a))
    return wavelet

def calculate_wavelet_transform(eeg_window, window, a, b, f_max):
    """Calculate the wavelet transform using the provided parameters."""
    return integrate.trapz(eeg_window * create_wavelet(window, a, b, f_max), window)

def generate_wavelet_matrix(eeg, k_dis, n_a, f_max, a_mass, morlet_half_period):
    """Generate a matrix containing wavelet transform values."""
    n = len(eeg)
    c_mass = [[0] * n for _ in range(n_a)]

    for j in range(morlet_half_period, n - morlet_half_period):
        st = j - morlet_half_period
        en = j + morlet_half_period
        window = np.linspace(st / k_dis, en / k_dis, en - st)
        eeg_window = eeg[st:en]  # Обрабатываем массив eeg здесь
        b = [window[int(len(window) / 2)] for _ in range(len(window))]  # Определяем b здесь
        for i in range(n_a):
            a = a_mass[i]
            c_mass[i][j] = calculate_wavelet_transform(eeg_window, window, a, b, f_max)
    return c_mass

def compute_instant_power(c_mass, a_mass, f_research, f_max):
    """Compute the instantaneous power within the desired range."""
    n = len(c_mass[0])
    instant_power = [0 for _ in range(n)]
    for i in range(len(c_mass)):
        if f_research[0] <= f_max / a_mass[i] <= f_research[1]:
            for j in range(len(c_mass[i])):
                instant_power[j] += c_mass[i - 1][j]
    return instant_power

def compute_averaged_power(instant_power, absence_half_period, morlet_half_period, k_dis):
    """Compute the averaged power within the desired range."""
    n = len(instant_power)
    averaged_power = [0 for _ in range(n)]
    for i in range(absence_half_period + morlet_half_period, n - absence_half_period - morlet_half_period):
        averaged_power[i] = (integrate.trapz(instant_power[i - absence_half_period:i + absence_half_period], dx=1/k_dis)) / (2 * absence_half_period / k_dis)
    return averaged_power

def compute_standard_deviation(instant_power, absence_half_period, morlet_half_period):
    """Compute the standard deviation within the desired range."""
    n = len(instant_power)
    sd = [0 for _ in range(n)]
    for i in range(absence_half_period + morlet_half_period, n - absence_half_period - morlet_half_period):
        sd[i] = np.std(instant_power[i - absence_half_period:i + absence_half_period])
    return sd