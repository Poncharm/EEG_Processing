#@title Calc. func.
import mne
import numpy as np
from scipy import integrate

def load_and_filter_raw_data(file_path, l_freq=1.0, h_freq=50.0):
    """Load raw data from the specified file and apply bandpass filter."""
    raw = mne.io.read_raw_edf(file_path, preload=True, encoding='latin1')
    # Apply bandpass filter
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw


def change_data_range(channel_data, start_index, end_index):
    """Filter the desired range of data and convert to microvolts."""
    return channel_data[start_index:end_index] * (10 ** 6)


def create_wavelet(window, a, b):
    """Create a Morlet wavelet based on the provided parameters."""
    wavelet = (1 / a ** 0.5) * np.exp(-((((window - b) / a) ** 2) / 2)) * (np.cos(2 * np.pi * F_MAX * (window - b) / a))
    return wavelet


def calculate_wavelet_transform(eeg_window, window, a, b):
    """Calculate the wavelet transform using the provided parameters."""
    return integrate.trapz(eeg_window * create_wavelet(window, a, b), window)


def generate_wavelet_matrix(eeg):
    """Generate a matrix containing wavelet transform values."""
    c_mass = [[0] * N for i in range(N_A)]

    for j in range(MORLET_HALF_PERIOD, N - MORLET_HALF_PERIOD):
        st = j - MORLET_HALF_PERIOD
        en = j + MORLET_HALF_PERIOD
        window = T[st:en]
        eeg_window = eeg[st:en]  # Обрабатываем массив eeg здесь
        b = [T[j] for _ in range(len(window))]  # Определяем b здесь
        for i in range(N_A):
            a = A_MASS[i]
            c_mass[i][j] = calculate_wavelet_transform(eeg_window, window, a, b)
    return c_mass


def compute_instant_power(c_mass):
    """Compute the instantaneous power within the desired range."""
    instant_power = [0 for i in range(N)]
    for i in range(len(c_mass)):
        if F_MAX / A_MASS[i] >= F_RESEARCH[0] and F_MAX / A_MASS[i] <= F_RESEARCH[1]:
            for j in range(len(c_mass[i])):
                instant_power[j] += c_mass[i - 1][j]
    return instant_power


def compute_averaged_power(instant_power):
    """Compute the averaged power within the desired range."""
    averaged_power = [0 for i in range(N)]
    for i in range(ABSENCE_HALF_PERIOD + MORLET_HALF_PERIOD, N - ABSENCE_HALF_PERIOD - MORLET_HALF_PERIOD):
        for j in range(ABSENCE_HALF_PERIOD):
            averaged_power[i] = (integrate.trapz(instant_power[i - ABSENCE_HALF_PERIOD:i + ABSENCE_HALF_PERIOD],
                                        T[i - ABSENCE_HALF_PERIOD:i + ABSENCE_HALF_PERIOD])) / (ABSENCE_HALF_PERIOD * 2 / K_DIS)
    return averaged_power

def compute_standard_deviation(instant_power):
    """Compute the standart deviation within the desired range."""
    sd = [0 for i in range(N)]
    for i in range(ABSENCE_HALF_PERIOD + MORLET_HALF_PERIOD, N - ABSENCE_HALF_PERIOD - MORLET_HALF_PERIOD):
      sd[i] = np.std(instant_power[i-ABSENCE_HALF_PERIOD:i+ABSENCE_HALF_PERIOD])
    return sd


def binarize_data(averaged_power, sd):
    """Convert data into a binary form."""
    binar = [0 for i in range(N)]
    sequence_indices = []  # list to hold start and end indices of sequences of 1s
    for i in range(N):
        if (averaged_power[i] >= W_MIN and averaged_power[i] <= W_MAX) and (sd[i] >= SD_MIN and sd[i] <= SD_MAX):
            binar[i] = 1

    count = 0  # count of consecutive 1s
    start = 0  # start index of sequence of 1s
    for i in range(N):
        if binar[i] == 1:
            if count == 0:  # new sequence of 1s starts
                start = i
            count += 1
        elif count > 0:  # sequence of 1s ended
            sequence_indices.append((start, i-1))  # save start and end indices
            count = 0  # reset count
    if count > 0:  # check if there is a sequence of 1s at the end
        sequence_indices.append((start, N-1))  # save start and end indices

    # If there is a start of an absence but no end, assume the absence lasts till the end of the data
    if len(sequence_indices) > 0 and len(sequence_indices[-1]) == 1:
        sequence_indices[-1] = (sequence_indices[-1][0], N-1)

    return binar, sequence_indices
