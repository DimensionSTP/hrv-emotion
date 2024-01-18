import os
from typing import List

import pandas as pd
from tqdm import tqdm


class Augmentation():
    def __init__(
        self,
        information_df_path : str,
        info_columns: int,
        repeat: int,
        data_path: str,
        emotions: List[str],
        save_path: str,
        sheet_name: str,
    ) -> None:
        self.information_df = pd.read_excel(information_df_path)
        self.info_columns = info_columns
        self.repeat = repeat
        self.data_path = data_path
        self.emotions = emotions
        self.save_path = save_path
        self.sheet_name = sheet_name

    def __call__(self) -> None:
        stratched_information_df = self.stratch_information_df()
        augmented_df = self.get_augmented_df(stratched_information_df)
        augmented_df.to_excel(
            self.save_path, 
            sheet_name=self.sheet_name,
            index=False,
        )

    def stratch_information_df(self) -> pd.DataFrame:
        information_df = self.information_df.iloc[:, :self.info_columns]
        stratched_information_df = information_df.reindex(information_df.index.repeat(self.repeat)).reset_index(drop=True)
        return stratched_information_df

    def get_augmented_df(self, stratched_information_df: pd.DataFrame,) -> pd.DataFrame:
        hrv_values_dfs = []
        emotion_folder_paths = [f"{self.data_path}/{emotion}" for emotion in self.emotions]
        for folder_path in emotion_folder_paths:
            for file_name in tqdm(os.listdir(folder_path)):
                file_path = f"{folder_path}/{file_name}"
                data_df = pd.read_csv(file_path)
                data_df = data_df.iloc[:self.repeat]
                hrv_values_dfs.append(data_df)

        hrv_df = pd.concat(hrv_values_dfs).reset_index(drop=True)
        augmented_df = pd.concat([stratched_information_df, hrv_df], axis=1)
        return augmented_df
