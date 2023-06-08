import numpy as np
import time

from Processing_Fucs import *
from Plot_Funcs import *

# Define constants and parameters
# chb24
FILES = [[r'Data_eeg\1.24 (The only work)\chb24_01.edf', [484, 498], [2455, 2468]],
         [r'Data_eeg\1.24 (The only work)\chb24_03.edf', [235, 246], [2888, 2900]],
         [r'Data_eeg\1.24 (The only work)\chb24_04.edf', [1094, 1108], [1417, 1428], [1750, 1759]],
         [r'Data_eeg\1.24 (The only work)\chb24_06.edf', [1233, 1243]],
         [r'Data_eeg\1.24 (The only work)\chb24_07.edf', [42, 53]],
         [r'Data_eeg\1.24 (The only work)\chb24_09.edf', [1749.2, 1759]],
         [r'Data_eeg\1.24 (The only work)\chb24_11.edf', [3582, 3590]],
         [r'Data_eeg\1.24 (The only work)\chb24_13.edf', [3291, 3300]],
         [r'Data_eeg\1.24 (The only work)\chb24_14.edf', [1942, 1951]],
         [r'Data_eeg\1.24 (The only work)\chb24_15.edf', [3556, 3565]],
         [r'Data_eeg\1.24 (The only work)\chb24_17.edf', [3518, 3528]],
         [r'Data_eeg\1.24 (The only work)\chb24_21.edf', [2807, 2819]]]

# Pt-10, Pt-12
# FILES = [['EEG_Pt-10.edf', [7, 21]],
#          ['EEG_Pt-12.edf', [8.325, 12]]]

# REQUIRED_CHANNELS = ["EEG Fp1-REF", "EEG Fp2-REF", "EEG T7-REF", "EEG T8-REF", "EEG O1-REF", "EEG O2-REF"]

K_DIS = 125
N_A = 20  # Number of scales for wavelet transform

F_RESEARCH = [2, 4]  # Investigated frequencies
F_MIN = F_RESEARCH[0]
F_MAX = F_RESEARCH[1]
F_MASS = np.linspace(F_MIN, F_MAX, N_A)  # Frequency range to build the graph
A_MASS = [F_MAX / F_MASS[i] for i in range(N_A)]  # Mass of A-values

ABSENCE_HALF_PERIOD = round((1 / F_RESEARCH[0] / 2) * K_DIS)  # Half period for the averaging window
MORLET_HALF_PERIOD = round(K_DIS / 3 / 2) * 4  # Delay for the wavelet transform


def main(channel):
    # Load and filter data
    eeg = change_data_range(raw_data[channel], TN_MIN, TN_MAX)

    # Generate wavelet matrix
    c_mass = generate_wavelet_matrix(eeg)

    # Compute instant power and averaged power
    instant_power = compute_instant_power(np.abs(c_mass))
    averaged_power = compute_averaged_power(instant_power)

    # Define index range for data extraction
    index_range = slice(ABSENCE_HALF_PERIOD + MORLET_HALF_PERIOD, -ABSENCE_HALF_PERIOD - MORLET_HALF_PERIOD)

    # Extract mean power within the range
    mean_power = np.mean(averaged_power[index_range])

    # Extract min and max power within the range
    w_min = min(averaged_power[index_range])
    w_max = max(averaged_power[index_range])

    # Compute sliding standard deviation
    instant_power_to_sd = compute_instant_power(c_mass)
    sd = compute_standard_deviation(instant_power_to_sd)

    # Extract mean standard deviation within the range
    mean_sd = np.mean(sd[index_range])

    # Extract min and max standard deviation within the range
    sd_min = min(sd[index_range])
    sd_max = max(sd[index_range])

    return w_min, w_max, mean_power, sd_min, sd_max, mean_sd


if __name__ == "__main__":
    start_time = time.time()

    # Loop over files and their respective seizure timestamps
    for file_data in FILES:
        FILE = file_data[0]
        timestamps = file_data[1:]

        # Load data and resample it
        raw = load_and_filter_raw_data(FILE, 1, 50)
        raw_resampled = raw.copy().resample(sfreq=K_DIS)
        raw_data = raw_resampled.get_data()

        # Print channel names for debugging
        channel_names = raw_resampled.ch_names
        print(f"CHANNEL_NAMES: {channel_names}")
        print()

        # Identify indices of required channels
        required_channel_indices = [channel_names.index(ch) for ch in REQUIRED_CHANNELS if ch in channel_names]

        # Processing of each seizure timestamp
        print(f"Processed file: {FILE}")
        print()
        for timestamp in timestamps:
            T_ST_ABSENCE = timestamp[0]
            T_EN_ABSENCE = timestamp[1]
            T_MIN = T_ST_ABSENCE - (ABSENCE_HALF_PERIOD + MORLET_HALF_PERIOD) / K_DIS  # Start time-pont (second)
            T_MAX = T_EN_ABSENCE + (ABSENCE_HALF_PERIOD + MORLET_HALF_PERIOD) / K_DIS  # End time-pont (second)

            TN_MIN = round(K_DIS * T_MIN)  # Start index of the data range
            TN_MAX = round(K_DIS * T_MAX)  # End index of the data range
            N = TN_MAX - TN_MIN  # Numbers of points
            T = np.linspace(T_MIN, T_MAX, N)  # Time vector

            W_MAX = 0
            W_MIN = 0
            SD_MAX = 0
            SD_MIN = 0
            MEAN_POWER = 0
            MEAN_SD = 0

            print(f"Time start: {T_ST_ABSENCE}, Time end: {T_EN_ABSENCE}")
            print()

            # Processing of each channel
            for i in required_channel_indices:
                w_min, w_max, mean_power, sd_min, sd_max, mean_sd = main(channel=i)
                MEAN_POWER += mean_power
                MEAN_SD += mean_sd
                W_MIN += w_min
                W_MAX += w_max
                SD_MIN += sd_min
                SD_MAX += sd_max
                # print(f"Channel {channel_names[i]} is processed")

            # print()
            print("FINAL RESULTS:")
            print(f"W_MIN: {round(W_MIN / 6, 2)}, W_MAX: {round(W_MAX / 6, 2)}, Mean power: {round(MEAN_POWER / 6, 2)}")
            print(f"SD_MIN: {round(SD_MIN / 6, 2)}, SD_MAX: {round(SD_MAX / 6, 2)}, Mean SD: {round(MEAN_SD / 6, 2)}")
            print()

    end_time = time.time()
    total_time = end_time - start_time
    print("Время выполнения программы: {:.2f} секунд".format(total_time))
