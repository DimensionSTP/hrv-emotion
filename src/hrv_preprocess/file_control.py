import os
import shutil
from typing import List

import pandas as pd
from tqdm import tqdm


class FileController():
    def __init__(
        self,
        all_files_path: str,
        emotions_raw_path: str,
        information_df_path : str,
        info_columns: int,
        template_data_path: str,
        template_data_name: str,
        emotions: List[str],
        template_save_path: str,
        template_sheet_name: str,
    ) -> None:
        self.all_files_path = all_files_path
        self.emotions_raw_path = emotions_raw_path
        self.information_df_path = information_df_path
        self.info_columns = info_columns
        self.template_data_path = template_data_path
        self.template_data_name = template_data_name
        self.emotions = emotions
        self.template_save_path = template_save_path
        self.template_sheet_name = template_sheet_name
    
    def move_files_per_emotions(self) -> None:
        raw_hrv_files = os.listdir(self.all_files_path)
        for raw_hrv_file in tqdm(raw_hrv_files):
            raw_hrv_file_path = f"{self.all_files_path}/{raw_hrv_file}"
            emotion = raw_hrv_file.split("_")[1]
            save_path = f"{self.emotions_raw_path}/{emotion.lower()}"
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            
            if emotion == "HAHV":
                shutil.copy(raw_hrv_file_path, f"{save_path}/{raw_hrv_file}")
            elif emotion == "HALV":
                shutil.copy(raw_hrv_file_path, f"{save_path}/{raw_hrv_file}")
            elif emotion == "LAHV":
                shutil.copy(raw_hrv_file_path, f"{save_path}/{raw_hrv_file}")
            elif emotion == "LALV":
                shutil.copy(raw_hrv_file_path, f"{save_path}/{raw_hrv_file}")
            elif emotion == "reference":
                shutil.copy(raw_hrv_file_path, f"{save_path}/{raw_hrv_file}")
            else:
                raise ValueError("Invalid emotion")

    def make_test_template(self) -> None:
        information_df = pd.read_excel(self.information_df_path)
        information_df = information_df.iloc[:, :self.info_columns]
        template_data_dfs = []
        for emotion in tqdm(self.emotions):
            df = pd.read_csv(f"{self.template_data_path}/{self.template_data_name}_{emotion}.csv")
            template_data_dfs.append(df)
        data_df = pd.concat(template_data_dfs).reset_index(drop=True)
        test_template = pd.concat([information_df, data_df], axis=1)
        test_template.to_excel(
            self.template_save_path, 
            sheet_name=self.template_sheet_name,
            index=False,
        )
