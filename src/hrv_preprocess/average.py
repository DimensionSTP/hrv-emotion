import os
from typing import List

import pandas as pd
from tqdm import tqdm


class Average():
    def __init__(
        self,
        data_path: str,
        emotions: List[str],
        save_path: str,
        save_name: str,
    ) -> None:
        self.data_path = data_path
        self.emotions = emotions
        self.save_path = save_path
        self.save_name = save_name

    def __call__(self) -> None:
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        for emotion in self.emotions:
            emotion_data_path = f"{self.data_path}/{emotion}"
            emotion_save_name = f"{self.save_name}_{emotion}"
            self.get_average_hrv(emotion_data_path, emotion_save_name)

    def get_average_hrv(
        self, 
        emotion_data_path: str, 
        emotion_save_name: str,
    ) -> None:
        data_file_names = os.listdir(emotion_data_path)
        for i, data_file_name in enumerate(tqdm(data_file_names)):
            data_file = pd.read_csv(f"{emotion_data_path}/{data_file_name}")
            average_data_file = data_file.mean()
            if  i == 0:
                average_df = average_data_file
            else:
                average_df = pd.concat([average_df, average_data_file], axis=1)
        average_df.T.to_csv(
            f"{self.save_path}/{emotion_save_name}.csv", 
            encoding="utf-8-sig", 
            index=False,
        )
