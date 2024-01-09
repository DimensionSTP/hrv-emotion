import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def interpolate_double(
    original_signal_path: str,
    interpolated_signal_path: str,
) -> None:
    if not os.path.exists(interpolated_signal_path):
        os.makedirs(interpolated_signal_path, exist_ok=True)

    original_signal_files = os.listdir(original_signal_path)
    for signal_file in tqdm(original_signal_files):
        original_signal = pd.read_csv(f"{original_signal_path}/{signal_file}")
        original_num_samples = len(original_signal)
        interpolated_num_samples = original_num_samples * 2
        original_signal.index = np.arange(0, 2 * original_num_samples, step=2)
        interpolated_signal = original_signal.reindex(range(interpolated_num_samples)).interpolate(method="linear")
        interpolated_signal.to_csv(f"{interpolated_signal_path}/{signal_file}", index=False)


if __name__ == "__main__":
    ORIGINAL_SIGNAL_PATHS = ["./dataset/sl_rppg/signals", "./dataset/sl_rppg_60/signals", "./dataset/sl_rppg_120/signals", "./dataset/sl_rppg_240/signals"]
    INTERPOLATED_SIGNAL_PATHS = ["./dataset/sl_rppg_60/signals", "./dataset/sl_rppg_120/signals", "./dataset/sl_rppg_240/signals", "./dataset/sl_rppg_480/signals"]
    for i in range(len(ORIGINAL_SIGNAL_PATHS)):
        interpolate_double(
            original_signal_path=ORIGINAL_SIGNAL_PATHS[i],
            interpolated_signal_path=INTERPOLATED_SIGNAL_PATHS[i],
        )