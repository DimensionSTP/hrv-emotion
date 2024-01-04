import os

import pandas as pd


def normalize_hrv(emotion_path, reference_path, save_path):
    emotion_files = os.listdir(emotion_path)
    reference_files = os.listdir(reference_path)
    if len(emotion_files) == len(reference_files):
        for i in range(len(emotion_files)):
            emotion_data = pd.read_csv(f"{emotion_path}/{emotion_files[i]}")
            reference_data = pd.read_csv(f"{reference_path}/{reference_files[i]}")
            reference_mean_values = reference_data.mean()
            normalized_data = (emotion_data - reference_mean_values) / reference_mean_values
            if emotion_files[i][:3] == reference_files[i][:3]:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                normalized_data.to_csv(f"{save_path}/{emotion_files[i][:-4]}_normalized.csv", index=False)
            else:
                raise Exception("emotion file subject name doesn't match with reference file subject name.")
    else:
        raise Exception("number of emotion files doesn't match with reference files.")


if __name__ == "__main__":
    BASIC_PATH = "./dataset/sl/emotions_ppg_210_1/raw"
    EMOTION_FOLDER_PATHS = ["hahv", "halv", "lahv", "lalv"]
    REFERENCE_PATH = "reference"
    SAVE_FOLDER_PATH = "./dataset/sl/emotions_ppg_210_1/solidly_normalized"
    
    reference_path = f"{BASIC_PATH}/{REFERENCE_PATH}"
    for emotion_folder_path in EMOTION_FOLDER_PATHS:
        emotion_path = f"{BASIC_PATH}/{emotion_folder_path}"
        save_path = f"{SAVE_FOLDER_PATH}/{emotion_folder_path}"
        normalize_hrv(emotion_path, reference_path, save_path)