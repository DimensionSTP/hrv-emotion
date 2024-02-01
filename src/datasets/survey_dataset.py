from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

import pandas as pd


class LabDataset():
    def __init__(
        self,
        df_path: str,
        meaningful_features: List[str], 
        emotion: str, 
        standard: str, 
        condition: str, 
        filter_outlier: bool,
    ) -> None:
        self.meaningful_features = meaningful_features
        self.df_path = df_path
        self.emotion = emotion
        self.standard = standard
        self.condition = condition
        self.filter_outlier = filter_outlier

    def __call__(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = self.get_survey_dataset()
        return dataset

    def get_survey_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_excel(self.df_path).iloc[:, 1:]
        if self.emotion == "stress":
            if self.standard == "compound":
                if self.condition == "extreme":
                    high = df[(df["Arousal"] >= 6) & (df["자극_Arousal"] == 1) & (df["Valence"] <= 2) & (df["자극_Valence"] == 0)]
                    low = df[(df["Arousal"] <= 2) & (df["자극_Arousal"] == 0) & (df["Valence"] >= 6) & (df["자극_Valence"] == 1)]
                elif self.condition == "normal":
                    high = df[(df["Arousal"] >= 5) & (df["자극_Arousal"] == 1) & (df["Valence"] <= 3) & (df["자극_Valence"] == 0)]
                    low = df[(df["Arousal"] <= 3) & (df["자극_Arousal"] == 0) & (df["Valence"] >= 5) & (df["자극_Valence"] == 1)]
                else:
                    raise ValueError("Invalid condition")
            elif self.standard == "survey":
                if self.condition == "extreme":
                    high = df[(df["Arousal"] >= 6) & (df["Valence"] <= 2)]
                    low = df[(df["Arousal"] <= 2) & (df["Valence"] >= 6)]
                elif self.condition == "normal":
                    high = df[(df["Arousal"] >= 5) & (df["Valence"] <= 3)]
                    low = df[(df["Arousal"] <= 3) & (df["Valence"] >= 5)]
                else:
                    raise ValueError("Invalid condition")
            else:
                raise ValueError("Invalid standard")
        elif self.emotion == "arousal":
            if self.standard == "compound":
                if self.condition == "extreme":
                    high = df[(df["Arousal"] >= 6) & (df["자극_Arousal"] == 1)]
                    low = df[(df["Arousal"] <= 2) & (df["자극_Arousal"] == 0)]
                elif self.condition == "normal":
                    high = df[(df["Arousal"] >= 5) & (df["자극_Arousal"] == 1)]
                    low = df[(df["Arousal"] <= 3) & (df["자극_Arousal"] == 0)]
                else:
                    raise ValueError("Invalid condition")
            elif self.standard == "survey":
                if self.condition == "extreme":
                    high = df[(df["Arousal"] >= 6)]
                    low = df[(df["Arousal"] <= 2)]
                elif self.condition == "normal":
                    high = df[(df["Arousal"] >= 5)]
                    low = df[(df["Arousal"] <= 3)]
                else:
                    raise ValueError("Invalid condition")
            else:
                raise ValueError("Invalid standard")
        elif self.emotion == "valence":
            if self.standard == "compound":
                if self.condition == "extreme":
                    high = df[(df["Valence"] >= 6) & (df["자극_Valence"] == 1)]
                    low = df[(df["Valence"] <= 2) & (df["자극_Valence"] == 0)]
                elif self.condition == "normal":
                    high = df[(df["Valence"] >= 5) & (df["자극_Valence"] == 1)]
                    low = df[(df["Valence"] <= 3) & (df["자극_Valence"] == 0)]
                else:
                    raise ValueError("Invalid condition")
            elif self.standard == "survey":
                if self.condition == "extreme":
                    high = df[(df["Valence"] >= 6)]
                    low = df[(df["Valence"] <= 2)]
                elif self.condition == "normal":
                    high = df[(df["Valence"] >= 5)]
                    low = df[(df["Valence"] <= 3)]
                else:
                    raise ValueError("Invalid condition")
            else:
                raise ValueError("Invalid standard")
        else:
            raise ValueError("Invalid emotion")

        if self.filter_outlier ==  True:
            groups = [high, low]
            filtered_groups = []
            num_group_outliers = []
            for group in groups:
                group_mean_values = group.mean()
                group_std_values = group.std()
                group["outlier"] = ((group > group_mean_values + 3 * group_std_values) | (group < group_mean_values - 3 * group_std_values)).any(axis=1).astype(int)
                num_group_outlier = len(group.loc[group["outlier"] == 1])
                group = group[(group["outlier"] == 0)]
                filtered_groups.append(group)
                num_group_outliers.append(num_group_outlier)
            high = filtered_groups[0]
            low = filtered_groups[1]
        elif self.filter_outlier == False:
            num_group_outliers = [0, 0]
            high = high
            low = low
        else:
            raise ValueError("Invalid filter_outlier")

        high["label"] = 1
        low["label"] = 0
        dataset = pd.concat([high, low], ignore_index=True)

        data = dataset[self.meaningful_features]
        label = dataset["label"]
        return (data, label)
