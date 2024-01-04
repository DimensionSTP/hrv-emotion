import numpy as np
import numpy.fft as fft
from scipy import interpolate
from scipy.signal import savgol_filter
from iir import IIR

class HRV:
    def __init__(self):
        self.iir = IIR()
        self.labels = [
            "Time", 
            "BPM", 
            "SDNN", 
            "rMSSD", 
            "pNN50", 
            "VLF", 
            "LF", 
            "HF", 
            "VLFp", 
            "LFp", 
            "HFp", 
            "lnVLF", 
            "lnLF", 
            "lnHF", 
            "VLF/HF", 
            "LF/HF", 
            "tPow", 
            "dPow", 
            "dHz", 
            "pPow", 
            "pHz", 
            "CohRatio", 
            "RSA_PB"
        ]
    
    # HRV (Time + Frequency domain RSA_PB)
    def __call__(self, peak, sampling_rate):
        """
        ppi  = ms
        """
        before_ppi = self.calc_ppi(peak=peak, sampling_rate=sampling_rate)
        ppi = self.ppi_correction(ppi=before_ppi)
        # df (re-sampling: 2 Hz)
        dt = 0.5
        fs = 1 / dt
        df = fs / len(ppi)
        
        # fft
        power_spectrum = self.calc_fft(ppi)
        
        # time domain
        mean_ppi = np.average(ppi)
        bpm = round(self.calc_bpm(ppi), 1)
        if (np.isnan(bpm)) | (np.isinf(bpm)): # exception
            return [np.nan] * 21
        sdnn = self.calc_sdnn(ppi)
        rmssd = self.calc_rmssd(ppi)
        pnn50 = self.calc_pnn50(ppi)
        rsa_pb = self.calc_rsa(ppi)
        
        # frequency domain
        vlf, lf, hf = self.calc_f(power_spectrum, df)
        vlf_p, lf_p, hf_p = self.calc_f_p(vlf, lf, hf)
        ln_vlf, ln_lf, ln_hf = self.calc_ln_f(vlf, lf, hf)
        vlf_hf, lf_hf = self.calc_per_hf(vlf, lf, hf)
        
        t_pow = self.calc_t_pow(vlf, lf, hf)
        d_pow, d_hz = self.calc_d(power_spectrum, df)
        p_pow, p_hz = self.calc_p(power_spectrum, df)
        coh_ratio = self.calc_coh_ratio(t_pow, p_pow)
        
        return [
            mean_ppi, 
            bpm, 
            sdnn, 
            rmssd, 
            pnn50, 
            vlf, 
            lf, 
            hf, 
            vlf_p, 
            lf_p, 
            hf_p, 
            ln_vlf, 
            ln_lf, 
            ln_hf, 
            vlf_hf, 
            lf_hf, 
            t_pow, 
            d_pow, 
            d_hz, 
            p_pow, 
            p_hz, 
            coh_ratio, 
            rsa_pb
        ]
    
    # PPI (= RRI, IBI(interbeat interval))
    def calc_ppi(self, peak, sampling_rate):
        ppi = np.diff(peak) / sampling_rate * 1000  # ppi를 ms로 변환
        return ppi # array
    
    # PPI correction
    def ppi_correction(self, ppi):
        _nni = []
        nni = []
        for i in range(len(ppi)):
            if (ppi[i] >= 60000.0 / 240) and (ppi[i] < 60000.0 / 40):  # 30 ~ 180 bpm Normal range 벗어나면 버림
                _nni.append(ppi[i])
            else:
                pass
        """
            # nni_mean : normal range ppi 평균값
        nni_mean = np.nanmean(_nni)
        
        min = nni_mean * 0.7
        max = nni_mean * 1.3
        
        for k in range(len(_nni)):
            if (_nni[k] <= min) or (_nni[k] >= max):  # 전체 평균의 30%보다 작거나 130%보다 크면 이전 정상 nni의 4개값으로 대체
                nni.append(nni_mean)
            else:
                nni.append(_nni[k])
        """
        
        for k in range(len(_nni)):
            if k < 4:  # 처음 4개까지는 그대로 정상 nni 취급
                nni.append(_nni[k])
                
            else:
                mean_b4 = np.nanmean(_nni[k - 4: k - 1])
                # min_b4 = mean_b4 - 2 * np.std(_nni[k - 4: k - 1])
                # max_b4 = mean_b4 + 2 * np.std(_nni[k - 4: k - 1])
                
                if (_nni[k] < mean_b4 * 0.8) or (
                        _nni[k] > mean_b4 * 1.3):  # 이전 ppi 4개 평균의 30%보다 작거나 130%보다 크면 이전 정상 nni의 4개 평균값으로 대체
                    _nni[k] = mean_b4
                    nni.append(_nni[k])
                    
                else:
                    nni.append(_nni[k])  # 위의 조건에 해당하지 않으면 원래값 그대로
        # print(len(nni))
        return nni
    
    # Time domain
    def calc_bpm(self, ppi):
        """
        :param ppi: ms
        :return:
        """
        ppi = np.trim_zeros(ppi)
        # for _ in range(ppi.count(0)):
        #     ppi.remove(0)
        avg_ppi = np.average(ppi)
        bpm = 60000.0 / avg_ppi
        return bpm
    
    def calc_sdnn(self, ppi):
        """
        SDNN is a standard deviation of the NN intervals, which is the square root of their variance.
        :param ppi: Peak to Peak Intervals
        :return: SDNN
        """
        sdnn = np.std(ppi, ddof=1)  # ddof = n-1로 나눠줌
        return sdnn
    
    def calc_rmssd(self, ppi):
        """
        rMSSD is the square root of the mean squared differences of successive NN intervals.
        :param ppi: Peak to Peak Intervals
        :return: rMSSD
        """
        diff_ppi = np.diff(ppi)
        rmssd = np.sqrt(np.average(diff_ppi **2))
        return rmssd
    
    def calc_pnn50(self, ppi):
        """
        pNN50 is NN50 count divided by the total number of all NN intervals.
        NN50 count is number of pairs of adjacent NN intervals differing by more than 50 ms in the entire recording.
        :param ppi: Peak to Peak Intervals
        :return: pNN50: 단위는 이미 %로 변환
        """
        diff_ppi = np.diff(ppi)
        nn50 = np.sum(np.abs(diff_ppi)>50)
        pnn50 = nn50 / len(ppi) * 100            # percent 변환
        return pnn50
    
    def calc_rsa(self, ppi, epoch= 30, interval=30):
        epoch_lnvar_list = []
        rsa_pb = 0
        # # 2Hz resampling (2Hz resampling이 안되었다면 이걸 살려도 됨)
        # secs = int(len(ecg) / 500)  # Number of seconds in signal X  (ppi는 샘플링 수를 정해줘야하는 신호이기 떄문에 시간을 알려면 ecg 신호로 대체해서 하는수밖에 없음)
        # samps = secs * 2  # Number of samples to downsample
        # resampled = resample(ppi, samps)
        
        # Savitzky Golay filter (Moving polynomial)
        savitzky = savgol_filter(ppi, 21, 3)  # 21 point filter, 3rd order
        
        # resampled ppi - savitzky
        series = ppi - savitzky
        
        # chebyshev bandpass filter ( band range: 0.12-0.4Hz,  sampling rate = 2Hz)
        filtered_cheby1 = self.iir.cheby1_bandpass_filter(series, 0.12, 0.4, 2)
        
        # log transformation & averaging  3epochs (1 epoch = 30sec))
        for i in range(0, len(filtered_cheby1) - epoch, interval):
            sliding = filtered_cheby1[i:i + epoch]
            epoch_var = np.var(sliding)
            epoch_lnvar = abs(np.log(epoch_var))
            epoch_lnvar_list.append(epoch_lnvar)
            rsa_pb = np.mean(epoch_lnvar_list)
        return rsa_pb
    
    # Frequency domain
    def calc_fft(self, signal):
        # fft
        frequency = np.array(fft.fft(signal))
        # power spectrum
        power_spectrum = ((frequency.real * frequency.real) + (frequency.imag * frequency.imag)) / (len(frequency) * len(frequency)) * 2
        power_spectrum = power_spectrum[:int(len(power_spectrum)/2)]
        return power_spectrum
    
    def calc_f(self, power_spectrum, df):
        """
        Very Low Frequency (VLF) is a band of power spectrum range between 0.0033 and 0.04 Hz.
        :param power_spectrum: Power spectrum calcd from PPI by FFT.
        :return: VLF
        """
        vlf_index1 = int(0.0033 / df) + 1
        vlf_index2 = int(0.04 / df)
        if vlf_index1 <= 1:
            vlf_index1 = 2
        try:
            vlf_band = power_spectrum[vlf_index1:vlf_index2+1]
            vlf = np.sum(vlf_band)
        except:
            vlf = 0
    
        """
        Low Frequency (LF) is a band of power spectrum range between 0.04 and 0.15 Hz.
        :param power_spectrum: Power spectrum calcd from PPI by FFT.
        :return: LF
        """
        lf_index1 = int(0.04 / df) + 1
        lf_index2 = int(0.15 / df)
        try:
            lf_band = power_spectrum[lf_index1:lf_index2+1]
            lf = np.sum(lf_band)
        except:
            lf = 0
    
        """
        High Frequency (HF) is a band of power spectrum range between 0.15 and 0.4 Hz.
        :param power_spectrum: Power spectrum calcd from PPI by FFT.
        :return: hF
        """
        hf_index1 = int(0.15 / df) + 1
        hf_index2 = int(0.4 / df)
        try:
            hf_band = power_spectrum[hf_index1:hf_index2+1]
            hf = np.sum(hf_band)
        except:
            hf = 0
        return vlf, lf, hf
    
    def calc_f_p(self, vlf, lf, hf):
        total_power = vlf + lf + hf
        vlf_p = vlf / total_power
        lf_p = lf / total_power
        hf_p = hf / total_power
        return vlf_p, lf_p, hf_p
    
    def calc_ln_f(self, vlf, lf, hf):
        if vlf <= 0:
            ln_vlf = 0
        else: 
            ln_vlf = abs(np.log(vlf))
            
        if lf <= 0:
            ln_lf = 0
        else: 
            ln_lf = abs(np.log(lf))
            
        if hf <= 0:
            ln_hf = 0
        else: 
            ln_hf = abs(np.log(hf))
        return ln_vlf, ln_lf, ln_hf
    
    def calc_per_hf(self, vlf, lf, hf):
        vlf_hf = vlf / hf
        lf_hf = lf / hf
        return vlf_hf, lf_hf
    
    def calc_t_pow(self, vlf, lf, hf):
        total_power = vlf + lf + hf
        return total_power
    
    def calc_d(self, power_spectrum, df):
        # find highest peak
        d_power = np.max(power_spectrum[2:])
        # find highest hz
        d_hz = (2 + np.argmax(power_spectrum[2:])) * df
        return d_power, d_hz
    
    def calc_p(self, power_spectrum, df):
        # split 0.04 ~ 0.26 hz
        index1 = int(0.04 / df) + 1
        index2 = int(0.26 / df)
        
        band = power_spectrum[index1:index2+1]
        # find highest peak
        peak_index = index1 + np.argmax(band)
        # split interval hz
        interval_index = int(0.015 / df)
        peak = power_spectrum[peak_index-interval_index:peak_index+interval_index+1]
        # calc total power of the peak
        peak_power = np.sum(peak)
        
        power_spectrum = power_spectrum[index1:index2+1]
        # find highest hz
        peak_hz = (index1 + np.argmax(power_spectrum)) * df
        return peak_power, peak_hz
    
    def calc_coh_ratio(self, total_power, peak_power):
        coh_ratio = pow(peak_power / (total_power - peak_power), 2)
        return coh_ratio
    
    def cubic_spline_interpolation(self, signal, sampling_rate, time):
        x = np.linspace(0, 1, num=len(signal))
        tck = interpolate.splrep(x, signal, s=0)
        xnew = np.linspace(0, 1, num=sampling_rate*time)
        return interpolate.splev(xnew, tck, der=0)