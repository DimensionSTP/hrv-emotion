from typing import Tuple, Union

import numpy as np
from scipy.signal import butter, lfilter, cheby1


class IIR:
    def __init__(self) -> None:
        pass
    
    def filtered_signal(self, filter: np.ndarray,) -> np.ndarray:
        filtered_signal = []
        for i in range(0, len(filter)):
            filtered_signal.append(float(filter[i]))
        return np.array(filtered_signal)
    
    def butter_bandpass(
        self, 
        lowcut: Union[float, int], 
        highcut: Union[float, int], 
        fs: int, 
        order: int = 5,
    ) -> Tuple[np.ndarray]:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return (b, a)


    def butter_bandpass_filter(
        self, 
        data: np.ndarray, 
        lowcut: Union[float, int], 
        highcut: Union[float, int], 
        fs: int, 
        order: int = 5,
    ) -> np.ndarray:
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    def butter_lowpass(
        self, 
        cut: Union[float, int], 
        fs: int, 
        order: int = 5,
    ) -> Tuple[np.ndarray]:
        nyq = 0.5 * fs
        ncut = cut / nyq
        b, a = butter(order, ncut, btype="low", analog=True)
        return (b, a)

    def butter_lowpass_filter(
        self, 
        data: np.ndarray, 
        cut: Union[float, int], 
        fs: int, 
        order: int = 5,
    ) -> np.ndarray:
        b, a = self.butter_lowpass(cut, fs, order=order)
        y = lfilter(b, a, data)
        return self.filtered_signal(y)

    def butter_highpass(
        self, 
        cut: Union[float, int], 
        fs: int, 
        order: int = 5,
    ) -> Tuple[np.ndarray]:
        nyq = 0.5 * fs
        ncut = cut / nyq
        b, a = butter(order, ncut, btype="high", analog=True)
        return (b, a)

    def butter_highpass_filter(
        self, 
        data: np.ndarray, 
        cut: Union[float, int], 
        fs: int, 
        order: int = 5,
    ) -> np.ndarray:
        b, a = self.butter_highpass(cut, fs, order=order)
        y = lfilter(b, a, data)
        return self.filtered_signal(y)

    ## Chebychev type 1 bandpass filter
    """
    data = ecg or ppg (type:1d array)
    lowcut= minimum frequency (Hz)
    highcut= maximum frequency (Hz)
    fs = sampling rate
    output = filtered ecg or ppg (type:1d array)
    """
    def cheby1_bandpass(
        self, 
        lowcut: Union[float, int], 
        highcut: Union[float, int], 
        fs: int,
    ) -> Tuple[np.ndarray]:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = cheby1(3, 3, [low, high], btype="band")
        return (b, a)

    def cheby1_bandpass_filter(
        self, 
        data: np.ndarray, 
        lowcut: Union[float, int], 
        highcut: Union[float, int], 
        fs: int,
    ) -> np.ndarray:
        b, a = self.cheby1_bandpass(lowcut, highcut, fs)
        y = lfilter(b, a, data)
        return self.filtered_signal(y)
