import numpy as np
import matplotlib.pyplot as plt


def plot_data(axis, data, title='', fontsize=10, color="salmon"):
    axis.set_title(title, fontsize=fontsize)
    axis.grid(which='both', axis='both', linestyle='--')
    axis.plot(data, color=color, zorder=1)


def plot_points(axis, values, indices):
    axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)


def moving_max(signal, window_size):
    trend = []
    half_window_size = int(window_size / 2)

    # Add first component
    trend.append(signal[0])
    # Add middle components
    for i in range(1, half_window_size):
        trend.append(np.max(signal[0:i]))
    for i in range(half_window_size, len(signal)-half_window_size):
        trend.append(np.max(signal[i-half_window_size:i+half_window_size]))
    for i in range(len(signal)-half_window_size, len(signal)-1):
        trend.append(np.max(signal[i:]))
    # Add last component
    trend.append(signal[-1])

    #
    # plt.plot(signal)
    # plt.plot(trend)
    # plt.show()
    #
    return trend


def find_peaks_threshold(signal, threshold=800):
    detected_peaks_indices = []
    for i in range(len(signal)):
        if signal[i] >= threshold:
            detected_peaks_indices.append(i)

    # plt.plot(detected_peaks_indices)
    # plt.show()
    return detected_peaks_indices


def detect_peaks(signal, fs=500, detrend_factor=1, is_display=False, filename='A'):
    # Estimate trend
    raw_signal = signal
    trend = moving_max(signal, window_size=int(fs*detrend_factor))
    # De-trend
    # signal = np.array(signal) - np.array(trend)
    signal = np.array(signal) - trend

    # Find peaks
    detected_peaks_indices = find_peaks_threshold(signal, threshold=0)

    # Display
    if is_display:
        fig, axarr = plt.subplots(4, sharex=True, figsize=(15, 12))
        plot_data(axis=axarr[0], data=raw_signal, title='Raw Signal')
        plot_data(axis=axarr[0], data=trend, title='Raw Signal', color='red')
        plot_data(axis=axarr[1], data=signal, title='Detrend Signal')
        plot_data(axis=axarr[2], data=raw_signal, title='Raw Signal with peaks marked (black) by Maxima')
        plot_points(axis=axarr[2], values=raw_signal, indices=detected_peaks_indices)
        plot_data(axis=axarr[3], data=signal, title='Detrend Signal with peaks marked (black)')
        plot_points(axis=axarr[3], values=signal, indices=detected_peaks_indices)
        plt.tight_layout()
        fig.savefig(filename)
        plt.show()
        plt.clf()
        plt.close()

    return detected_peaks_indices
