import os

from tqdm import tqdm

def rename_sl_rppg_dataset_files_in_directory(ecg_dir: str, rppg_dir: str,) -> None:
    ecg_files = sorted(os.listdir(ecg_dir))
    rppg_files = sorted(os.listdir(rppg_dir), key=lambda x: int(x.split('test')[1].split('.')[0]))
    for i, rppg_file in enumerate(tqdm(rppg_files)):
        ecg_file = ecg_files[i]
        os.rename(os.path.join(rppg_dir, rppg_file), os.path.join(rppg_dir, ecg_file))


if __name__ == "__main__":
    ECG_DIR = "./dataset/sl/signals"
    RPPG_DIR = "./dataset/sl_rppg/signals"
    rename_sl_rppg_dataset_files_in_directory(ECG_DIR, RPPG_DIR)