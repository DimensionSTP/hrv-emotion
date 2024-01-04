import numpy as np
import pandas as pd

def show_describe(data_path):
    describe_df = pd.DataFrame()
    df = pd.read_csv(data_path)
    for column in df.columns:
        describe = df[column].describe()
        describe_per_feature = pd.DataFrame(describe, columns = [column]).T
        describe_per_feature["std_error"] = np.nan
        describe_per_feature["std_error"] = describe_per_feature["std"] / np.sqrt(describe_per_feature["count"])
        describe_per_feature.drop(["count"], axis=1, inplace=True)
        describe_per_feature = describe_per_feature[["mean", "std", "std_error", "min", "25%", "50%", "75%", "max"]]
        describe_per_feature.columns = ["평균", "표준 편차", "표준 오차", "최솟값", "25%", "중간값", "75%", "최댓값"]
        describe_df = describe_df.append(describe_per_feature)
    return describe_df


if __name__ == "__main__":
    describe_df = show_describe("survey_normalized_low_valence_compound_extreme.csv")
    describe_df.to_excel(
        excel_writer="survey_normalized_low_valence_compound_extreme_describe.xlsx", 
        sheet_name="low_valence"
    )