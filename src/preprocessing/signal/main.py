import os
from typing import List

import pandas as pd

from tqdm import tqdm

from hrv import HRV
import maxima_peak_detection as mpd


hrv = HRV()

def main(
    window_size: int,
    interval_size: int,
    sampling_rate: int,
    data_path: str,
    file_name: List,
    save_path: str,
):
    for file in tqdm(file_name):
        # ppg_df = pd.read_csv(
        #     f"{data_path}/{file}", header=None, delimiter="\t", skiprows=23
        # )
        # ppg_df = pd.read_csv(f"{data_path}/{file}")
        ppg_df = pd.read_csv(f"{data_path}/{file}", header=None)
        # print(file + "분석 시작!")

        ppg_value = ppg_df[1].values
        # ppg_value = ppg_df[2].values
        # ppg_value = ppg_df[0].values
        # Sliding window
        hrv_list = []

        for i in range(
            0,
            len(ppg_value) - window_size * sampling_rate,
            interval_size * sampling_rate,
        ):
            sliding_ppg = ppg_value[i : i + window_size * sampling_rate]
            sliding_ppg = sliding_ppg.astype("float")
            # Peak detection
            peak = mpd.detect_peaks(
                sliding_ppg, fs=sampling_rate, detrend_factor=1, is_display=False
            )
            # Calcualte HRV
            hrv_list.append(hrv(peak=peak, sampling_rate=sampling_rate))
        # file write (.csv)
        df = pd.DataFrame(hrv_list)
        df.columns = [
            "RRI",
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
            "RSA_PB",
        ]
        df.to_csv(
            f"{save_path}/{file[:-4]}_{window_size}.csv",
            encoding="utf-8-sig",
            index=False,
        )  # 결과 저장


if __name__ == "__main__":
    WINDOW_SIZE = 210
    INTERVAL_SIZE = 1
    SAMPLING_RATE = 500
    DATA_PATH = "./dataset/sl/ecg"
    FILE_NAME = os.listdir(DATA_PATH)
    SAVE_PATH = "./dataset/sl/data_ppg_210_1"
    main(
        window_size=WINDOW_SIZE,
        interval_size=INTERVAL_SIZE,
        sampling_rate=SAMPLING_RATE,
        data_path=DATA_PATH,
        file_name=FILE_NAME,
        save_path=SAVE_PATH,
        )