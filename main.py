import numpy as np
import time

from Processing_Fucs import *
from Plot_Funcs import *

file = open(r'D:\PythonProjects\EEG_Processing\Result_Data', "w+")

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

REQUIRED_CHANNELS = ["FP1-F7"]

K_DIS = 125
N_A = 20  # Number of scales for wavelet transform

F_RESEARCH = [2, 4]  # Investigated frequencies
F_MIN = F_RESEARCH[0]
F_MAX = F_RESEARCH[1]
F_MASS = np.linspace(F_MIN, F_MAX, N_A)  # Frequency range to build the graph
A_MASS = [F_MAX / F_MASS[i] for i in range(N_A)]  # Mass of A-values

ABSENCE_HALF_PERIOD = round(K_DIS / F_RESEARCH[0] / 2)  # Half period for the averaging window
MORLET_HALF_PERIOD = round(K_DIS / 3 / 2) * 4  # Delay for the wavelet transform


def main(channel_index, raw_data, TN_MIN, TN_MAX, K_DIS, N_A, F_MAX, A_MASS, ABSENCE_HALF_PERIOD, MORLET_HALF_PERIOD):
    # Load and filter data
    eeg = change_data_range(raw_data[channel_index], TN_MIN, TN_MAX)

    # Generate wavelet matrix
    c_mass = generate_wavelet_matrix(eeg, K_DIS, N_A, F_MAX, A_MASS, MORLET_HALF_PERIOD)

    # Compute instant power and averaged power
    instant_power = compute_instant_power(np.abs(c_mass), A_MASS, F_RESEARCH, F_MAX)
    averaged_power = compute_averaged_power(instant_power, ABSENCE_HALF_PERIOD, MORLET_HALF_PERIOD, K_DIS)

    # Define index range for data extraction
    index_range = slice(ABSENCE_HALF_PERIOD + MORLET_HALF_PERIOD, - ABSENCE_HALF_PERIOD - MORLET_HALF_PERIOD)

    # Extract mean power within the range
    mean_power = np.mean(averaged_power[index_range])

    # Extract min and max power within the range
    w_min = min(averaged_power[index_range])
    w_max = max(averaged_power[index_range])

    # Compute sliding standard deviation
    instant_power_to_sd = compute_instant_power(c_mass, A_MASS, F_RESEARCH, F_MAX)
    sd = compute_standard_deviation(instant_power_to_sd, ABSENCE_HALF_PERIOD, MORLET_HALF_PERIOD)

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
        print(channel_names)

        for start, end in timestamps:
            print(f'timestamp: {start}-{end} c')
            # Convert timestamps to indices
            tn_min = round(start * K_DIS - (ABSENCE_HALF_PERIOD+MORLET_HALF_PERIOD))
            tn_max = round(end * K_DIS + (ABSENCE_HALF_PERIOD+MORLET_HALF_PERIOD))

            w_min_res = 0
            w_max_res = 0
            for channel in REQUIRED_CHANNELS:
                if channel in channel_names:
                    # Get channel index
                    channel_index = channel_names.index(channel)

                    # Call main processing function
                    w_min, w_max, mean_power, sd_min, sd_max, mean_sd = main(channel_index, raw_data, tn_min,
                                                                             tn_max, K_DIS, N_A, F_MAX, A_MASS, ABSENCE_HALF_PERIOD, MORLET_HALF_PERIOD)

                    print(w_min, w_max, mean_power, sd_min, sd_max, mean_sd)
                    w_min_res += w_min
                    w_max_res += w_max
            print(f'{round(w_min_res, 2)}, {round(w_max_res, 2)}')
            file.write(f'timestamp: {start}-{end} c: ')
            file.write(f'{round(w_min_res, 2)}, {round(w_max_res, 2)}')
            file.write('\n')
            file.write('\n')

    end_time = time.time()
    total_time = end_time - start_time
    file.write("Время выполнения программы: {:.2f} секунд".format(total_time))
    file.close()