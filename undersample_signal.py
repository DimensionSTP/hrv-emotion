import os

import pandas as pd
from tqdm import tqdm


def undersample_signal(
    original_sampling_rate: int,
    desired_sampling_rate: int,
    original_signal_path: str,
    undersampled_signal_path: str,
) -> None:
    interval = round(original_sampling_rate / desired_sampling_rate)
    
    if not os.path.exists(undersampled_signal_path):
        os.makedirs(undersampled_signal_path, exist_ok=True)
    
    original_signal_files = os.listdir(original_signal_path)
    for signal_file in tqdm(original_signal_files):
        original_signal = pd.read_csv(f"{original_signal_path}/{signal_file}")
        undersampled_signal = original_signal.iloc[::interval]
        undersampled_signal.to_csv(f"{undersampled_signal_path}/{signal_file}", index=False)


if __name__ == "__main__":
    ORIGINAL_SAMPLING_RATE = 500
    DESIRED_SAMPLING_RATES = [250, 100, 50]
    ORIGINAL_SIGNAL_PATH = "./dataset/sl/signals"
    UNDERSAMPLED_DATASET_PATH = "./dataset/sl_"
    for desired_sampling_rate in DESIRED_SAMPLING_RATES:
        undersample_signal(
            original_sampling_rate=ORIGINAL_SAMPLING_RATE,
            desired_sampling_rate=desired_sampling_rate,
            original_signal_path=ORIGINAL_SIGNAL_PATH,
            undersampled_signal_path=f"{UNDERSAMPLED_DATASET_PATH}{desired_sampling_rate}/signals",
        )
