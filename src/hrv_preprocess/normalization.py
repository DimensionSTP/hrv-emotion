import os
from typing import List

import pandas as pd
from tqdm import tqdm


class Normalization():
    def __init__(
        self,
        raw_path: str,
        emotions: List[str],
        reference: str,
        save_path: str,
    ):
        self.raw_path = raw_path
        self.emotions = emotions
        self.reference = reference
        self.save_path = save_path

    def __call__(self) -> None:
        reference_path = f"{self.raw_path}/{self.reference}"
        for emotion in self.emotions:
            raw_emotion_path = f"{self.raw_path}/{emotion}"
            save_emotion_path = f"{self.save_path}/{emotion}"
            self.normalize_hrv(raw_emotion_path, reference_path, save_emotion_path)

    def normalize_hrv(
        self,
        raw_emotion_path: str,
        reference_path: str,
        save_emotion_path: str,
    ) -> None:
        emotion_files = os.listdir(raw_emotion_path)
        reference_files = os.listdir(reference_path)
        if len(emotion_files) == len(reference_files):
            for i in tqdm(range(len(emotion_files))):
                emotion_data = pd.read_csv(f"{raw_emotion_path}/{emotion_files[i]}")
                reference_data = pd.read_csv(f"{reference_path}/{reference_files[i]}")
                reference_mean_values = reference_data.mean()
                normalized_data = (emotion_data - reference_mean_values) / reference_mean_values
                if emotion_files[i][:3] == reference_files[i][:3]:
                    if not os.path.exists(save_emotion_path):
                        os.makedirs(save_emotion_path)
                    normalized_data.to_csv(f"{save_emotion_path}/{emotion_files[i][:-4]}_normalized.csv", index=False)
                else:
                    raise Exception("Emotion file subject name doesn't match with reference file subject name.")
        else:
            raise Exception("Number of emotion files doesn't match with reference files.")
