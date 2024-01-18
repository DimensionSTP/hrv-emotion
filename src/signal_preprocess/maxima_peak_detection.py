from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt


class MaximaPeakDetection():
    def __init__(
        self,
        threshold: int,
        detrend_factor: int,
        is_display: bool,
    ) -> None:
        self.threshold = threshold
        self.detrend_factor = detrend_factor
        self.is_display = is_display
        self.filename = "peak"

    def __call__(
        self,
        signal: np.ndarray,
        sampling_rate: int,
    ) -> List[float]:
        # Estimate trend
        raw_signal = signal
        trend = self.moving_max(signal, window_size=int(sampling_rate*self.detrend_factor))
        # De-trend
        signal = np.array(signal) - trend

        # Find peaks
        detected_peaks_indices = self.find_peaks_threshold(signal)

        # Display
        if self.is_display:
            fig, axarr = plt.subplots(4, sharex=True, figsize=(15, 12))
            self.plot_data(axis=axarr[0], data=raw_signal, title="Raw Signal")
            self.plot_data(axis=axarr[0], data=trend, title="Raw Signal", color="red")
            self.plot_data(axis=axarr[1], data=signal, title="Detrend Signal")
            self.plot_data(axis=axarr[2], data=raw_signal, title="Raw Signal with peaks marked (black) by Maxima")
            self.plot_points(axis=axarr[2], values=raw_signal, indices=detected_peaks_indices)
            self.plot_data(axis=axarr[3], data=signal, title="Detrend Signal with peaks marked (black)")
            self.plot_points(axis=axarr[3], values=signal, indices=detected_peaks_indices)
            plt.tight_layout()
            fig.savefig(self.filename)
            plt.show()
            plt.clf()
            plt.close()

        return detected_peaks_indices

    def moving_max(
        self, 
        signal: np.ndarray, 
        window_size: int,
    ) -> List[float]:
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
        return trend

    def find_peaks_threshold(self, signal: np.ndarray,) -> List[float]:
        detected_peaks_indices = []
        for i in range(len(signal)):
            if signal[i] >= self.threshold:
                detected_peaks_indices.append(i)
        return detected_peaks_indices

    def plot_data(
        self, 
        axis: str, 
        data: Union[np.ndarray, List[float]], 
        title: str = "", 
        fontsize: int = 10, 
        color: str = "salmon",
    ) -> None:
        axis.set_title(title, fontsize=fontsize)
        axis.grid(which="both", axis="both", linestyle="--")
        axis.plot(data, color=color, zorder=1)

    def plot_points(
        self, 
        axis: str, 
        values: np.ndarray, 
        indices: List[float],
    ) -> None:
        axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)
