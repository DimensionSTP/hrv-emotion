import numpy as np
import pandas as pd


def show_describe(
    data_path: str, 
    meaningful_features: list, 
    filter_outlier: bool,
    ):
    describe_df = pd.DataFrame()
    df = pd.read_excel(data_path)
    conditioned_df = df[(df["Arousal"] <= 3) & (df["자극_Arousal"]==0) & (df["Valence"] >= 5) & (df["자극_Valence"]==1)]
    
    if filter_outlier == True:
        conditioned_df_mean_values = conditioned_df.mean()
        conditioned_df_std_values = conditioned_df.std()
        conditioned_df["outlier"] = ((conditioned_df > conditioned_df_mean_values + 3 * conditioned_df_std_values) | (conditioned_df < conditioned_df_mean_values - 3 * conditioned_df_std_values)).any(axis=1).astype(int)
        conditioned_df = conditioned_df[(conditioned_df["outlier"]==0)]
    elif filter_outlier == False:
        conditioned_df = conditioned_df
    else:
        raise ValueError("Invalid filter_outlier")
    
    for feature in meaningful_features:
        describe = conditioned_df[feature].describe()
        describe_per_feature = pd.DataFrame(describe, columns = [feature]).T
        describe_per_feature["std_error"] = np.nan
        describe_per_feature["std_error"] = describe_per_feature["std"] / np.sqrt(describe_per_feature["count"])
        describe_per_feature.drop(["count"], axis=1, inplace=True)
        describe_per_feature = describe_per_feature[["mean", "std", "std_error", "min", "25%", "50%", "75%", "max"]]
        describe_per_feature.columns = ["평균", "표준 편차", "표준 오차", "최솟값", "25%", "중간값", "75%", "최댓값"]
        describe_df = describe_df.append(describe_per_feature)
    return describe_df


if __name__ == "__main__":
    MEANINGFUL_FEATURES = [
    "RRI", 
    "BPM", 
    "SDNN",  
    "rMSSD", 
    "LF",  
    "lnLF", 
    "lnHF", 
    "tPow", 
    "pPow", 
    "dPow", 
    "pHz", 
    "dHz", 
    "CohRatio",
    "RSA_PB", 
    ]
    describe_df = show_describe(
        data_path="survey_normalized_80.xlsx", 
        meaningful_features=MEANINGFUL_FEATURES, 
        filter_outlier=True
        )
    describe_df.to_excel(
        excel_writer="./describes/filtered/survey_normalized_80_lahv_compound_describe.xlsx", 
        sheet_name="describe"
    )