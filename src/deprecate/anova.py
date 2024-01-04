import pandas as pd
from scipy import stats

columns = [
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

ha_df = pd.read_csv("./stimulus_preprocessed/high_valence_1_normalized.csv")
la_df = pd.read_csv("./stimulus_preprocessed/low_valence_1_normalized.csv")
df = pd.concat([ha_df, la_df], ignore_index=True)
arousal = df[(df["lnLF"]>=0) & (df["lnHF"]>=0) & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]
relaxation = df[(df["lnLF"]<0) & (df["lnHF"]<0) & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]
neutral1 = df[(df["lnLF"]>=0) & (df["lnHF"]<0) & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]
neutral2 = df[(df["lnLF"]<0) & (df["lnHF"]>=0) & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]
neutral =  pd.concat([neutral1, neutral2], ignore_index=True)

f_values = []
p_values = []
for column in columns:
    arousal_values = arousal[column].values
    relaxation_values = relaxation[column].values
    neutral_values = neutral[column].values
    f_value, p_value = stats.f_oneway(arousal_values, relaxation_values, neutral_values)
    f_values.append(f_value)
    p_values.append(p_value)

for i, p_value in enumerate(p_values):
    if p_value<0.0167:
        print(f"p value of {columns[i]} is {p_value}")
        # print(f"f value of {columns[i]} is {f_value}")