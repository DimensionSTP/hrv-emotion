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

arousal = pd.read_csv("./stimulus_preprocessed/high_valence_1_raw.csv")
relaxation = pd.read_csv("./stimulus_preprocessed/low_valence_1_raw.csv")
neutral = pd.read_csv("./stimulus_preprocessed/reference_1_raw.csv")

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
    if p_value<0.05:
        print(f"p value of {columns[i]} is {p_value}")
        print(f"f value of {columns[i]} is {f_value}")
        
# for i, p_value in enumerate(p_values):
#     print(f"{p_value:.03f}")