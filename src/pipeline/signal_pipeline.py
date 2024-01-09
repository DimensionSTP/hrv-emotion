import os

import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from ..utils.signal_setup import SignalSetUp
from ..signal_preprocess.hrv import HRV


def get_hrv(config: DictConfig) -> None:
    signal_setup = SignalSetUp(config)
    hrv = HRV()
    maxima_peak_detection = signal_setup.get_maxima_peak_detection()
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    if config.dataset == "sl":
        biopac_data_slicer = signal_setup.get_biopac_data_slicer()
        biopac_data_slicer()
        signal_column = 1
    elif config.dataset == "thirty" or config.dataset == "sl_rppg":
        signal_column = 0
    else:
        raise ValueError("Unvalid dataset")

    file_names = os.listdir(config.data_path)
    for file in tqdm(file_names):
        signal_df = pd.read_csv(f"{config.data_path}/{file}", header=None)
        signal_values = signal_df[signal_column].values
        hrv_list = []

        for i in range(
            0,
            len(signal_values) - config.window_size * config.sampling_rate,
            config.interval_size * config.sampling_rate,
        ):
            sliding_ppg = signal_values[i : i + config.window_size * config.sampling_rate]
            sliding_ppg = sliding_ppg.astype("float")
            # Peak detection
            peak = maxima_peak_detection(signal=sliding_ppg, sampling_rate=config.sampling_rate)
            # Calcualte HRV
            hrv_list.append(hrv(peak=peak, sampling_rate=config.sampling_rate))
        # file write (.csv)
        df = pd.DataFrame(hrv_list)
        df.columns = config.hrv_features
        df.to_csv(
            f"{config.save_path}/{file[:-4]}_{config.window_size}.csv",
            encoding="utf-8-sig",
            index=False,
        )  # 결과 저장