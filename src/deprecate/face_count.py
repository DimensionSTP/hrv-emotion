import os

from tqdm import tqdm
import pandas as pd

load_path = "./face_filled"
types = ["stimulus", "survey"]
emotions = ["hahv", "halv", "lahv", "lalv", "reference"]
save_path = "./face_count/whole"

for type in types:
    for emotion in emotions:
        data_list = os.listdir(f"{load_path}/{type}/{emotion}")
        result = pd.DataFrame(
                index =range(0, 50), 
                columns=["subject", "low", "neutral", "high", "undetected", "total_original", "total_summation", "is_total_same", "mode_idx"]
            )
        for i, data in enumerate(tqdm(data_list)):
            
            df = pd.read_csv(f"{load_path}/{type}/{emotion}/{data}")
            # df = df.loc[1800:3599]

            subject = data[:3]
            low_valence_count = len(df.loc[df["valence"] == 0])
            neutral_count = len(df.loc[df["valence"] == 1])
            high_valence_count = len(df.loc[df["valence"] == 2])
            undetected_count = len(df.loc[df["valence"] == 3])
            total_count_from_df = df.valence.count()
            total_count_from_summation = low_valence_count + neutral_count + high_valence_count + undetected_count
            counts = [low_valence_count, neutral_count, high_valence_count, undetected_count]
            mode = max(counts)

            if total_count_from_df == total_count_from_summation:
                is_total_same = 1
            else:
                is_total_same = 0

            if mode == low_valence_count:
                mode_idx = 0
            elif mode == neutral_count:
                mode_idx = 1
            elif mode == high_valence_count:
                mode_idx = 2
            elif mode == undetected_count:
                mode_idx = 3
            else:
                raise Exception("mode doesn't match with any value")

            row = [subject, low_valence_count, neutral_count, high_valence_count, undetected_count, total_count_from_df, total_count_from_summation, is_total_same, mode_idx]
            result.loc[i] = row
        result.to_csv(
            f"{save_path}/{type}_{emotion}.csv", 
            encoding="utf-8-sig", 
            index=False,
        )