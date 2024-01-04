import os

import pandas as pd
from scipy import stats


def analyze_statistics(
    features: list,
    df_path: str,
    result_path: str,
    emotion: str,
    standard: str,
    condition: str,
    filter_outlier: bool,
    method: str,
):
    df = pd.read_excel(df_path)

    if emotion == "24":
        if standard == "compound":
            if condition == "extreme":
                high = df[
                    (df["Arousal"] >= 6)
                    & (df["자극_Arousal"] == 1)
                    & (df["Valence"] <= 2)
                    & (df["자극_Valence"] == 0)
                ]
                low = df[
                    (df["Arousal"] <= 2)
                    & (df["자극_Arousal"] == 0)
                    & (df["Valence"] >= 6)
                    & (df["자극_Valence"] == 1)
                ]
                neutral = df[
                    (df["Arousal"] <= 3)
                    & (df["Arousal"] >= 5)
                    & (df["Valence"] >= 3)
                    & (df["Valence"] <= 5)
                ]
            elif condition == "normal":
                high = df[
                    (df["Arousal"] >= 5)
                    & (df["자극_Arousal"] == 1)
                    & (df["Valence"] <= 3)
                    & (df["자극_Valence"] == 0)
                ]
                low = df[
                    (df["Arousal"] <= 3)
                    & (df["자극_Arousal"] == 0)
                    & (df["Valence"] >= 5)
                    & (df["자극_Valence"] == 1)
                ]
                neutral = df[(df["Arousal"] == 4) & (df["Valence"] == 4)]
            else:
                raise ValueError("Invalid condition")
        elif standard == "survey":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6) & (df["Valence"] <= 2)]
                low = df[(df["Arousal"] <= 2) & (df["Valence"] >= 6)]
                neutral = df[
                    (df["Arousal"] <= 3)
                    & (df["Arousal"] >= 5)
                    & (df["Valence"] >= 3)
                    & (df["Valence"] <= 5)
                ]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5) & (df["Valence"] <= 3)]
                low = df[(df["Arousal"] <= 3) & (df["Valence"] >= 5)]
                neutral = df[(df["Arousal"] == 4) & (df["Valence"] == 4)]
            else:
                raise ValueError("Invalid condition")
        else:
            raise ValueError("Invalid standard")
    elif emotion == "arousal":
        if standard == "compound":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6) & (df["자극_Arousal"] == 1)]
                low = df[(df["Arousal"] <= 2) & (df["자극_Arousal"] == 0)]
                neutral = df[(df["Arousal"] <= 3) & (df["Arousal"] >= 5)]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5) & (df["자극_Arousal"] == 1)]
                low = df[(df["Arousal"] <= 3) & (df["자극_Arousal"] == 0)]
                neutral = df[(df["Arousal"] == 4)]
            else:
                raise ValueError("Invalid condition")
        elif standard == "survey":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6)]
                low = df[(df["Arousal"] <= 2)]
                neutral = df[(df["Arousal"] <= 3) & (df["Arousal"] >= 5)]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5)]
                low = df[(df["Arousal"] <= 3)]
                neutral = df[(df["Arousal"] == 4)]
            else:
                raise ValueError("Invalid condition")
        else:
            raise ValueError("Invalid standard")
    elif emotion == "valence":
        if standard == "compound":
            if condition == "extreme":
                high = df[(df["Valence"] >= 6) & (df["자극_Valence"] == 1)]
                low = df[(df["Valence"] <= 2) & (df["자극_Valence"] == 0)]
                neutral = df[(df["Valence"] <= 3) & (df["Valence"] >= 5)]
            elif condition == "normal":
                high = df[(df["Valence"] >= 5) & (df["자극_Valence"] == 1)]
                low = df[(df["Valence"] <= 3) & (df["자극_Valence"] == 0)]
                neutral = df[(df["Valence"] == 4)]
            else:
                raise ValueError("Invalid condition")
        elif standard == "survey":
            if condition == "extreme":
                high = df[(df["Valence"] >= 6)]
                low = df[(df["Valence"] <= 2)]
                neutral = df[(df["Valence"] <= 3) & (df["Valence"] >= 5)]
            elif condition == "normal":
                high = df[(df["Valence"] >= 5)]
                low = df[(df["Valence"] <= 3)]
                neutral = df[(df["Valence"] == 4)]
            else:
                raise ValueError("Invalid condition")
        else:
            raise ValueError("Invalid standard")
    else:
        raise ValueError("Invalid emotion")

    if filter_outlier == True:
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
    elif filter_outlier == False:
        is_outlier_filterd = "False"
        num_group_outliers = [0, 0, 0]
        high = high
        low = low
        neutral = neutral
    else:
        raise ValueError("Invalid filter_outlier")

    num_high = len(high)
    num_low = len(low)
    num_neutral = len(neutral)
    print(len(high))
    print(len(low))
    print(len(neutral))

    t_values = []
    p_values = []
    f_values = []
    meaningful_features = []

    if method == "t_test":
        for feature in features:
            high_values = high[feature].values
            low_values = low[feature].values
            neutral_values = neutral[feature].values
            t_value, p_value = stats.ttest_ind(high_values, low_values)
            t_values.append(t_value)
            p_values.append(p_value)

        for i, p_value in enumerate(p_values):
            if p_value < 0.05:
                meaningful_features.append(features[i])
                print(f"p value of {features[i]} is {p_value}")

    if method == "anova" and num_neutral != 0:
        for feature in features:
            high_values = high[feature].values
            low_values = low[feature].values
            neutral_values = neutral[feature].values
            f_value, p_value = stats.f_oneway(high_values, low_values, neutral_values)
            f_values.append(f_value)
            p_values.append(p_value)

        for i, p_value in enumerate(p_values):
            if p_value < 0.0167:
                meaningful_features.append(features[i])
                print(f"p value of {features[i]} is {p_value}")

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

    if os.path.isfile(result_path):
        original_result_df = pd.read_csv(result_path)
        new_result_df = pd.concat([original_result_df, result_df], ignore_index=True)
        new_result_df.to_csv(
            result_path,
            encoding="utf-8-sig",
            index=False,
        )
    else:
        result_df.to_csv(
            result_path,
            encoding="utf-8-sig",
            index=False,
        )


if __name__ == "__main__":
    FEATURES = [
        "RRI",
        "BPM",
        "SDNN",
        "rMSSD",
        # "pNN50",
        "VLF",
        "LF",
        "HF",
        "VLFp",
        "LFp",
        "HFp",
        "lnVLF",
        "lnLF",
        "lnHF",
        "VLF/HF",
        "LF/HF",
        "tPow",
        "dPow",
        "dHz",
        "pPow",
        "pHz",
        "CohRatio",
        "RSA_PB",
    ]
    DF_PATH = "survey_normalized_80.xlsx"
    RESULT_PATH = "analyze_statistics_results_filtered_outlier.csv"
    EMOTIONS = ["24", "arousal", "valence"]
    STANDARDS = ["compound", "survey"]
    CONDITIONS = ["extreme", "normal"]
    METHODS = ["t_test", "anova"]

    for emotion in EMOTIONS:
        for standard in STANDARDS:
            for condition in CONDITIONS:
                for method in METHODS:
                    analyze_statistics(
                        features=FEATURES,
                        df_path=DF_PATH,
                        result_path=RESULT_PATH,
                        emotion=emotion,
                        standard=standard,
                        condition=condition,
                        filter_outlier=True,
                        method=method,
                    )
