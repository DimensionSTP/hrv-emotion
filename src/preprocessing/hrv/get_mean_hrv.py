import os

import pandas as pd

def get_mean_hrv(data_path, save_path, save_name):
    data_file_names = os.listdir(data_path)
    for i, data_file_name in enumerate(data_file_names):
        data_file = pd.read_csv(f"{data_path}/{data_file_name}")
        mean_data_file = data_file.mean()
        if  i == 0:
            mean_df = mean_data_file
        else:
            mean_df = pd.concat([mean_df, mean_data_file], axis=1)
    mean_df.T.to_csv(
        f"{save_path}/{save_name}.csv", 
        encoding="utf-8-sig", 
        index=False,
    )


if __name__ == "__main__":
    STIMULI_PATH = "./dataset/sl_rppg/emotions/solidly_normalized"
    SAVE_PATH = "./dataset/sl_rppg/stimulus_preprocessed"
    SAVE_NAME = "normalized_1"
    EMOTIONS = ["hahv", "halv", "lahv", "lalv"]
    for emotion in EMOTIONS:
        stimuli_path = f"{STIMULI_PATH}/{emotion}"
        save_name = f"{SAVE_NAME}_{emotion}"
        get_mean_hrv(
            stimuli_path, SAVE_PATH, save_name
            )