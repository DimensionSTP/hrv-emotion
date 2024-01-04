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

df = pd.read_excel("survey_normalized.xlsx")

arousal = df[(df["자극_Valence"]==1) & (df["Valence"]>=5) & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]
relaxation = df[(df["자극_Valence"]==0) & (df["Valence"]<=3) & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]
neutral = df[df["Valence"]==4 & (df["LF/HF"]<=3.45) & (df["LF/HF"]>=1.15)]

# arousal = arousal[columns]
# relaxation = relaxation[columns]
# neutral = neutral[columns]

f_values = []
p_values = []
for column in columns:
    arousal_values = arousal[column].values
    relaxation_values = relaxation[column].values
    neutral_values = neutral[column].values
    f_value, p_value = stats.f_oneway(arousal_values, relaxation_values, neutral_values)
    # f_value, p_value = stats.ttest_ind(arousal_values, relaxation_values)
    f_values.append(f_value)
    p_values.append(p_value)
    
for i, p_value in enumerate(p_values):
    if p_value<0.0167:
        print(f"p value of {columns[i]} is {p_value}")