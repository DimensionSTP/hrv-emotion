from typing import List, Tuple

import pandas as pd


class LabDataset():
    def __init__(
        self,
        df_path: str,
        meaningful_features: List[str],
        condition: str,
    ) -> None:
        self.df_path = df_path
        self.meaningful_features = meaningful_features
        self.condition = condition

    def __call__(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = self.get_stimulus_dataset()
        return dataset

    def get_stimulus_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_excel(self.df_path).iloc[:, 1:]
        if self.condition == "arousal":
            df["label"] = df["자극_Arousal"]
        elif self.condition == "valence":
            df["label"] = df["자극_Valence"]
        elif self.condition == "dimensional":
            df["label"] = df.apply(
                lambda row: 1
                if row["자극_Arousal"] == 1 and row["자극_Valence"] == 1
                else 2
                if row["자극_Arousal"] == 1 and row["자극_Valence"] == 0
                else 3
                if row["자극_Arousal"] == 0 and row["자극_Valence"] == 1
                else 4,
                axis=1,
            )
        else:
            raise ValueError("Invalid condition")
        data = df[self.meaningful_features]
        label = df["label"]
        return (data, label)
