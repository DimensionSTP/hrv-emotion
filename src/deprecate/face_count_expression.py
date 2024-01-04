import os

from tqdm import tqdm
import pandas as pd

load_path = "./face_filled_expression"
types = ["stimulus", "survey"]
emotions = ["hahv", "halv", "lahv", "lalv", "reference"]
save_path = "./face_count_expression/detailed"

for type in types:
    for emotion in emotions:
        data_list = os.listdir(f"{load_path}/{type}/{emotion}")
        result = pd.DataFrame(
                index =range(0, 50), 
                columns=["subject", "neutral", "joy", "sadness", "surprise", "fear", "disgust", "anger", "contempt", "undetected", "total_original", "total_summation", "is_total_same", "mode_idx1", "mode_idx2", "mode_idx3", "mode_idx4"]
            )
        for i, data in enumerate(tqdm(data_list)):
            
            df = pd.read_csv(f"{load_path}/{type}/{emotion}/{data}")

            subject = data[:3]
            neutral_count = len(df.loc[df["valence"] == 0])
            joy_count = len(df.loc[df["valence"] == 1])
            sadness_count = len(df.loc[df["valence"] == 2])
            surprise_count = len(df.loc[df["valence"] == 3])
            fear_count = len(df.loc[df["valence"] == 4])
            disgust_count = len(df.loc[df["valence"] == 5])
            anger_count = len(df.loc[df["valence"] == 6])
            contempt_count = len(df.loc[df["valence"] == 7])
            undetected_count = len(df.loc[df["valence"] == 8])
            total_count_from_df = df.valence.count()
            total_count_from_summation = neutral_count + joy_count + sadness_count + surprise_count + fear_count + disgust_count + anger_count + contempt_count + undetected_count
            counts1 = [neutral_count, joy_count, sadness_count, surprise_count, fear_count, disgust_count, anger_count, contempt_count, undetected_count]
            mode1 = max(counts1)
            counts2 = [neutral_count, joy_count, sadness_count, surprise_count, fear_count, disgust_count, anger_count, contempt_count]
            mode2 = max(counts2)
            counts3 = [neutral_count, joy_count, surprise_count, fear_count, disgust_count, anger_count, contempt_count]
            mode3 = max(counts3)

            if total_count_from_df == total_count_from_summation:
                is_total_same = 1
            else:
                is_total_same = 0

            if mode1 == neutral_count:
                mode_idx1 = 0
            elif mode1 == joy_count:
                mode_idx1 = 1
            elif mode1 == sadness_count:
                mode_idx1 = 2
            elif mode1 == surprise_count:
                mode_idx1 = 3
            elif mode1 == fear_count:
                mode_idx1 = 4
            elif mode1 == disgust_count:
                mode_idx1 = 5
            elif mode1 == anger_count:
                mode_idx1 = 6
            elif mode1 == contempt_count:
                mode_idx1 = 7
            elif mode1 == undetected_count:
                mode_idx1 = 8
            else:
                raise Exception("mode doesn't match with any value")
            
            if mode2 == neutral_count:
                mode_idx2 = 0
            elif mode2 == joy_count:
                mode_idx2 = 1
            elif mode2 == sadness_count:
                mode_idx2 = 2
            elif mode2 == surprise_count:
                mode_idx2 = 3
            elif mode2 == fear_count:
                mode_idx2 = 4
            elif mode2 == disgust_count:
                mode_idx2 = 5
            elif mode2 == anger_count:
                mode_idx2 = 6
            elif mode2 == contempt_count:
                mode_idx2 = 7
            else:
                raise Exception("mode doesn't match with any value")
            
            if mode3 == neutral_count:
                mode_idx3 = 0
            elif mode3 == joy_count:
                mode_idx3 = 1
            elif mode3 == surprise_count:
                mode_idx3 = 3
            elif mode3 == fear_count:
                mode_idx3 = 4
            elif mode3 == disgust_count:
                mode_idx3 = 5
            elif mode3 == anger_count:
                mode_idx3 = 6
            elif mode3 == contempt_count:
                mode_idx3 = 7
            else:
                raise Exception("mode doesn't match with any value")
            
            if mode_idx3 == 0:
                mode_idx4 = 1
            elif mode_idx3 == 1:
                mode_idx4 = 2
            elif mode_idx3 == 3:
                mode_idx4 = 1
            elif mode_idx3 == 4:
                mode_idx4 = 0
            elif mode_idx3 == 5:
                mode_idx4 = 0
            elif mode_idx3 == 6:
                mode_idx4 = 0
            elif mode_idx3 == 7:
                mode_idx4 = 2
            else:
                raise Exception("mode doesn't match with any value")

            row = [subject, neutral_count, joy_count, sadness_count, surprise_count, fear_count, disgust_count, anger_count, contempt_count, undetected_count, total_count_from_df, total_count_from_summation, is_total_same, mode_idx1, mode_idx2, mode_idx3, mode_idx4]
            result.loc[i] = row
        result.to_csv(
            f"{save_path}/{type}_{emotion}.csv", 
            encoding="utf-8-sig", 
            index=False,
        )