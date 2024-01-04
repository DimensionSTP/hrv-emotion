import pandas as pd
from scipy import stats

columns = [
    "RRI",
    "BPM",
    "SDNN",
    "rMSSD",
    "pNN50",
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

ha_df = pd.read_csv("./stimulus_preprocessed/high_arousal_1_normalized.csv")
la_df = pd.read_csv("./stimulus_preprocessed/low_arousal_1_normalized.csv")
df = pd.concat([ha_df, la_df], ignore_index=True)
arousal = df[(df["lnLF"]>=0) & (df["lnHF"]>=0)]
relaxation = df[(df["lnLF"]<0) & (df["lnHF"]<0)]
neutral1 = df[(df["lnLF"]>=0) & (df["lnHF"]<0)]
neutral2 = df[(df["lnLF"]<0) & (df["lnHF"]>=0)]
neutral =  pd.concat([neutral1, neutral2], ignore_index=True)

t_values = []
p_values = []
for column in columns:
    arousal_values = arousal[column].values
    relaxation_values = relaxation[column].values
    neutral_values = neutral[column].values
    t_value, p_value = stats.ttest_ind(arousal_values, relaxation_values)
    t_values.append(t_value)
    p_values.append(p_value)
    
# print(p_values)
for i, p_value in enumerate(p_values):
    if p_value<0.05:
        print(f"p value of {columns[i]} is {p_value}")