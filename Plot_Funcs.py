import matplotlib.pyplot as plt
import numpy as np

from Ones import *
from Processing_Fucs import *

def plot_results(eeg=np.zeros(100), c_mass=np.zeros(100), averaged_power=np.zeros(100), sd=np.zeros(100), binar=np.zeros(100)):
    """Plot the data using subplots."""
    # Создаем оси графика
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))

    # EEG
    axs[0].grid(True)
    axs[0].set_xlim(T_MIN, T_MAX)
    # axs[0].set_ylim(-600, 600)
    axs[0].set_ylabel('Сигнал ЭЭГ, мкВ')
    axs[0].plot(T, eeg, color='black')
    axs[0].axvline(x=T_ST, color='r', linestyle='--')
    axs[0].axvline(x=T_EN, color='r', linestyle='--')

    # WT
    axs[1].grid(True)
    axs[1].set_xlim(T_MIN, T_MAX)
    axs[1].set_ylabel('Частота, Гц')
    contourf_plot = axs[1].contourf(T, F_MASS, c_mass, N_A, cmap=plt.cm.jet) # hot, cool, jet, viridis, inferno, magma, plasma, nipy_spectral
    axs[1].axhline(y=4, color='black', linestyle='--')
    axs[1].axhline(y=2, color='black', linestyle='--')
    # axs[1].set_yticks([1, 2, 3, 4, 5])
    # axs[1].set_yticklabels(['1', '2', '3', '4', '5'])

    # Создание новых осей для colorbar на правом краю текущих осей
    colorbar_axes = fig.add_axes([0.93, 0.58, 0.02, 0.15])  # Позиция и размер осей colorbar
    fig.colorbar(contourf_plot, cax=colorbar_axes)
    colorbar_axes.set_ylabel('Мгн. мощность, мкВ²', labelpad=-60)

    # Threshold
    axs[2].grid(True)
    axs[2].set_xlim(T_MIN, T_MAX)
    axs[2].set_ylabel('Усреднённая\nмощность, мкВ²')
    axs[2].plot(T, averaged_power)
    axs[2].axvline(x=T_ST, color='r', linestyle='--')
    axs[2].axvline(x=T_EN, color='r', linestyle='--')
    axs[2].hlines(W_MIN, 0, T_MAX, '0', color='black')
    axs[2].hlines(W_MAX, 0, T_MAX, '0', color='black')
    axs[2].text(T_MAX, W_MIN, r'$w_{кр1}$', verticalalignment='bottom')
    axs[2].text(T_MAX, W_MAX, r'$w_{кр2}$', verticalalignment='bottom')

    # Standard deviation
    axs[3].grid(True)
    axs[3].set_xlim(T_MIN, T_MAX)
    axs[3].set_ylabel('Стандартное\nотклонение, мкВ')
    axs[3].plot(T, sd)
    axs[3].axvline(x=T_ST, color='r', linestyle='--')
    axs[3].axvline(x=T_EN, color='r', linestyle='--')
    axs[3].hlines(SD_MIN, 0, T_MAX, '0', color='black')
    axs[3].hlines(SD_MAX, 0, T_MAX, '0', color='black')
    axs[3].text(T_MAX, SD_MIN, r'$sd_{кр1}$', verticalalignment='bottom')
    axs[3].text(T_MAX, SD_MAX, r'$sd_{кр1}$', verticalalignment='bottom')

    # Binary
    axs[4].grid(True)
    axs[4].set_xlim(T_MIN, T_MAX)
    axs[4].set_ylabel('Детектирование')
    axs[4].set_xlabel("t, с")
    axs[4].plot(T, binar)
    axs[4].set_yticks([0, 1])
    axs[4].set_yticklabels(['нома', 'приступ'])

    plt.show()
