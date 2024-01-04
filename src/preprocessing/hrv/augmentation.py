import os

import pandas as pd


information_df = pd.read_excel("./tabular_dataset/survey_normalized_130.xlsx")
information_df = information_df.iloc[:, :5]
information_df = information_df.reindex(information_df.index.repeat(60)).reset_index(drop=True)

hrv_values_dfs = []
folder_paths = [
    "./dataset/sl/emotions/solidly_normalized/hahv",
    "./dataset/sl/emotions/solidly_normalized/halv",
    "./dataset/sl/emotions/solidly_normalized/lalv",
    "./dataset/sl/emotions/solidly_normalized/lahv",
    "./dataset/thirty/emotions/solidly_normalized/halv",
    "./dataset/thirty/emotions/solidly_normalized/hahv",
    "./dataset/thirty/emotions/solidly_normalized/lahv",
    "./dataset/thirty/emotions/solidly_normalized/lalv",
    "./dataset/sl_rppg/emotions/solidly_normalized/hahv",
    "./dataset/sl_rppg/emotions/solidly_normalized/halv",
    "./dataset/sl_rppg/emotions/solidly_normalized/lalv",
    "./dataset/sl_rppg/emotions/solidly_normalized/lahv",
]

for folder_path in folder_paths:
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        data_df = pd.read_csv(file_path)
        data_df = data_df.iloc[:60]
        hrv_values_dfs.append(data_df)
        
hrv_df = pd.concat(hrv_values_dfs).reset_index(drop=True)

augmented_df = pd.concat([information_df, hrv_df], axis=1)

augmented_df.to_excel(
    "./tabular_dataset/survey_solidly_normalized_130_total_augmented.xlsx", 
    sheet_name="solidly_normalized",
    index=False,
)