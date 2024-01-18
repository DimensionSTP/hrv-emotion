import os
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy import stats
from tqdm import tqdm


class StatisticalAnalysis():
    def __init__(
        self,
        df_path: str,
        features: List[str],
        emotions: List[str],
        standards: List[str],
        conditions: List[str],
        filter_outlier: List[bool],
        methods: List[str],
        result_path: str,
        result_name: str,
    ) -> None:
        self.df = pd.read_excel(df_path).iloc[:, 1:]
        self.features = features
        self.emotions = emotions
        self.standards = standards
        self.conditions = conditions
        self.filter_outlier = filter_outlier
        self.methods = methods
        self.result_path = result_path
        self.result_name = result_name

    def __call__(self) -> None:
        for filter in self.filter_outlier:
            for emotion in tqdm(self.emotions):
                for standard in self.standards:
                    for condition in self.conditions:
                        for method in self.methods:
                            high, low, neutral, is_outlier_filterd, num_group_outliers = self.get_groups(emotion, standard, condition, filter)
                            self.analyze_statistics(emotion, standard, condition, method, high, low, neutral, is_outlier_filterd, num_group_outliers)

    def get_groups(
        self,
        emotion: str,
        standard: str,
        condition: str,
        filter: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, List[int]]:
        if emotion == "stress":
            if standard == "compound":
                if condition == "extreme":
                    high = self.df[
                        (self.df["Arousal"] >= 6)
                        & (self.df["자극_Arousal"] == 1)
                        & (self.df["Valence"] <= 2)
                        & (self.df["자극_Valence"] == 0)
                    ]
                    low = self.df[
                        (self.df["Arousal"] <= 2)
                        & (self.df["자극_Arousal"] == 0)
                        & (self.df["Valence"] >= 6)
                        & (self.df["자극_Valence"] == 1)
                    ]
                    neutral = self.df[
                        (self.df["Arousal"] <= 3)
                        & (self.df["Arousal"] >= 5)
                        & (self.df["Valence"] >= 3)
                        & (self.df["Valence"] <= 5)
                    ]
                elif condition == "normal":
                    high = self.df[
                        (self.df["Arousal"] >= 5)
                        & (self.df["자극_Arousal"] == 1)
                        & (self.df["Valence"] <= 3)
                        & (self.df["자극_Valence"] == 0)
                    ]
                    low = self.df[
                        (self.df["Arousal"] <= 3)
                        & (self.df["자극_Arousal"] == 0)
                        & (self.df["Valence"] >= 5)
                        & (self.df["자극_Valence"] == 1)
                    ]
                    neutral = self.df[(self.df["Arousal"] == 4) & (self.df["Valence"] == 4)]
                else:
                    raise ValueError("Invalid condition")
            elif standard == "survey":
                if condition == "extreme":
                    high = self.df[(self.df["Arousal"] >= 6) & (self.df["Valence"] <= 2)]
                    low = self.df[(self.df["Arousal"] <= 2) & (self.df["Valence"] >= 6)]
                    neutral = self.df[
                        (self.df["Arousal"] <= 3)
                        & (self.df["Arousal"] >= 5)
                        & (self.df["Valence"] >= 3)
                        & (self.df["Valence"] <= 5)
                    ]
                elif condition == "normal":
                    high = self.df[(self.df["Arousal"] >= 5) & (self.df["Valence"] <= 3)]
                    low = self.df[(self.df["Arousal"] <= 3) & (self.df["Valence"] >= 5)]
                    neutral = self.df[(self.df["Arousal"] == 4) & (self.df["Valence"] == 4)]
                else:
                    raise ValueError("Invalid condition")
            else:
                raise ValueError("Invalid standard")
        elif emotion == "arousal":
            if standard == "compound":
                if condition == "extreme":
                    high = self.df[(self.df["Arousal"] >= 6) & (self.df["자극_Arousal"] == 1)]
                    low = self.df[(self.df["Arousal"] <= 2) & (self.df["자극_Arousal"] == 0)]
                    neutral = self.df[(self.df["Arousal"] <= 3) & (self.df["Arousal"] >= 5)]
                elif condition == "normal":
                    high = self.df[(self.df["Arousal"] >= 5) & (self.df["자극_Arousal"] == 1)]
                    low = self.df[(self.df["Arousal"] <= 3) & (self.df["자극_Arousal"] == 0)]
                    neutral = self.df[(self.df["Arousal"] == 4)]
                else:
                    raise ValueError("Invalid condition")
            elif standard == "survey":
                if condition == "extreme":
                    high = self.df[(self.df["Arousal"] >= 6)]
                    low = self.df[(self.df["Arousal"] <= 2)]
                    neutral = self.df[(self.df["Arousal"] <= 3) & (self.df["Arousal"] >= 5)]
                elif condition == "normal":
                    high = self.df[(self.df["Arousal"] >= 5)]
                    low = self.df[(self.df["Arousal"] <= 3)]
                    neutral = self.df[(self.df["Arousal"] == 4)]
                else:
                    raise ValueError("Invalid condition")
            else:
                raise ValueError("Invalid standard")
        elif emotion == "valence":
            if standard == "compound":
                if condition == "extreme":
                    high = self.df[(self.df["Valence"] >= 6) & (self.df["자극_Valence"] == 1)]
                    low = self.df[(self.df["Valence"] <= 2) & (self.df["자극_Valence"] == 0)]
                    neutral = self.df[(self.df["Valence"] <= 3) & (self.df["Valence"] >= 5)]
                elif condition == "normal":
                    high = self.df[(self.df["Valence"] >= 5) & (self.df["자극_Valence"] == 1)]
                    low = self.df[(self.df["Valence"] <= 3) & (self.df["자극_Valence"] == 0)]
                    neutral = self.df[(self.df["Valence"] == 4)]
                else:
                    raise ValueError("Invalid condition")
            elif standard == "survey":
                if condition == "extreme":
                    high = self.df[(self.df["Valence"] >= 6)]
                    low = self.df[(self.df["Valence"] <= 2)]
                    neutral = self.df[(self.df["Valence"] <= 3) & (self.df["Valence"] >= 5)]
                elif condition == "normal":
                    high = self.df[(self.df["Valence"] >= 5)]
                    low = self.df[(self.df["Valence"] <= 3)]
                    neutral = self.df[(self.df["Valence"] == 4)]
                else:
                    raise ValueError("Invalid condition")
            else:
                raise ValueError("Invalid standard")
        else:
            raise ValueError("Invalid emotion")

        if filter == True:
            is_outlier_filterd = "True"
            groups = [high, low, neutral]
            filtered_groups = []
            num_group_outliers = []
            for group in groups:
                group_mean_values = group.mean()
                group_std_values = group.std()
                group["outlier"] = (
                    (
                        (group > group_mean_values + 3 * group_std_values)
                        | (group < group_mean_values - 3 * group_std_values)
                    )
                    .any(axis=1)
                    .astype(int)
                )
                num_group_outlier = len(group.loc[group["outlier"] == 1])
                group = group[(group["outlier"] == 0)]
                filtered_groups.append(group)
                num_group_outliers.append(num_group_outlier)
            high = filtered_groups[0]
            low = filtered_groups[1]
            neutral = filtered_groups[2]
        elif filter == False:
            is_outlier_filterd = "False"
            num_group_outliers = [0, 0, 0]
            high = high
            low = low
            neutral = neutral
        else:
            raise ValueError("Invalid filter_outlier")
        return (high, low, neutral, is_outlier_filterd, num_group_outliers)

    def analyze_statistics(
        self,
        emotion: str,
        standard: str,
        condition: str,
        method: str,
        high: pd.DataFrame,
        low: pd.DataFrame,
        neutral: pd.DataFrame,
        is_outlier_filterd: str,
        num_group_outliers: List[int],
    ) -> None:
        num_high = len(high)
        num_low = len(low)
        num_neutral = len(neutral)

        t_values = []
        p_values = []
        f_values = []
        meaningful_features = []
        if method == "t_test":
            for feature in self.features:
                high_values = high[feature].values
                low_values = low[feature].values
                neutral_values = neutral[feature].values
                t_value, p_value = stats.ttest_ind(high_values, low_values)
                t_values.append(t_value)
                p_values.append(p_value)

            for i, p_value in enumerate(p_values):
                if p_value < 0.05:
                    meaningful_features.append(self.features[i])
        elif method == "anova" and num_neutral != 0:
            for feature in self.features:
                high_values = high[feature].values
                low_values = low[feature].values
                neutral_values = neutral[feature].values
                f_value, p_value = stats.f_oneway(high_values, low_values, neutral_values)
                f_values.append(f_value)
                p_values.append(p_value)

            for i, p_value in enumerate(p_values):
                if p_value < 0.0167:
                    meaningful_features.append(self.features[i])
        elif method == "anova" and num_neutral == 0:
            pass
        else:
            raise ValueError("Invalid method")

        result = {
            "분류 감성": emotion,
            "그룹 분류 기준": standard,
            "주관 평가 경계": condition,
            "유의한 HRV 지표": meaningful_features,
            "분석 방법": method,
            "high 샘플 수": num_high,
            "low 샘플 수": num_low,
            "neutral 샘플 수": num_neutral,
            "이상치 제거 여부": is_outlier_filterd,
            "high 이상치 샘플 수": num_group_outliers[0],
            "low 이상치 샘플 수": num_group_outliers[1],
            "neutral 이상치 샘플 수": num_group_outliers[2],
        }
        result_df = pd.DataFrame.from_dict(result, orient="index").T

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)

        result_file = f"{self.result_path}/{self.result_name}"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat([original_result_df, result_df], ignore_index=True)
            new_result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
        else:
            result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
