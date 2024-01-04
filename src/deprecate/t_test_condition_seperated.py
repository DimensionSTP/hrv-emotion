import pandas as pd
from scipy import stats

columns = [
    "lnLF",
    "lnHF",
]

ha_df = pd.read_csv("./survey_preprocessed/halv_1_normalized.csv")
la_df = pd.read_csv("./survey_preprocessed/lahv_1_normalized.csv")

ha_arousal = ha_df[(ha_df["lnLF"]>=0) & (ha_df["lnHF"]>=0)]
ha_relaxation = ha_df[(ha_df["lnLF"]<0) & (ha_df["lnHF"]<0)]
ha_neutral1 = ha_df[(ha_df["lnLF"]>=0) & (ha_df["lnHF"]<0)]
ha_neutral2 = ha_df[(ha_df["lnLF"]<0) & (ha_df["lnHF"]>=0)]
ha_neutral =  pd.concat([ha_neutral1, ha_neutral2], ignore_index=True)

la_arousal = la_df[(la_df["lnLF"]>=0) & (la_df["lnHF"]>=0)]
la_relaxation = la_df[(la_df["lnLF"]<0) & (la_df["lnHF"]<0)]
la_neutral1 = la_df[(la_df["lnLF"]>=0) & (la_df["lnHF"]<0)]
la_neutral2 = la_df[(la_df["lnLF"]<0) & (la_df["lnHF"]>=0)]
la_neutral =  pd.concat([la_neutral1, la_neutral2], ignore_index=True)

ha_dfs = [ha_arousal, ha_relaxation, ha_neutral1, ha_neutral2, ha_neutral]
la_dfs = [la_arousal, la_relaxation, la_neutral1, la_neutral2, la_neutral]

for i in range(5):
    t_values = []
    p_values = []
    for column in columns:
        ha_df = ha_dfs[i][column].values
        la_df = la_dfs[i][column].values
        t_value, p_value = stats.ttest_ind(ha_df, la_df)
        t_values.append(t_value)
        p_values.append(p_value)
        print(f"p value of {column} is {p_value}")
        
    # for i, p_value in enumerate(p_values):
    #     if p_value<0.05:
    #         print(f"p value of {columns[i]} is {p_value}")