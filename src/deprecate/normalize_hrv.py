import os

import pandas as pd

EMOTION_PATH = "./additional_data/emotions/raw/lahv"
REFERENCE_PATH = "./additional_data/emotions/raw/reference"
EMOTION_FILES = os.listdir(EMOTION_PATH)
REFERENCE_FILES = os.listdir(REFERENCE_PATH)
SAVE_PATH = "./additional_data/emotions/normalized/lahv"

def normalize_hrv(emotion_path, reference_path, emotion_files, reference_files, save_path):
    for i in range(30):
        emotion_data = pd.read_csv(f"{emotion_path}/{emotion_files[i]}")
        reference_data = pd.read_csv(f"{reference_path}/{reference_files[i]}")
        sub_data = emotion_data.sub(reference_data)
        normalized_data = sub_data.div(reference_data)
        if emotion_files[i][:3] == reference_files[i][:3]:
            normalized_data.to_csv(f"{save_path}/{emotion_files[i][:-4]}_normalized.csv", index=False)
        else:
            raise Exception("emotion file subject name doesn't match with reference file subject name.")
        
if __name__ == "__main__":
    normalize_hrv(EMOTION_PATH, REFERENCE_PATH, EMOTION_FILES, REFERENCE_FILES, SAVE_PATH)