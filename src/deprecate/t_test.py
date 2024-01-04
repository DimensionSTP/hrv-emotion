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

ha_df = pd.read_csv("./stimulus_preprocessed/high_arousal_1_normalized.csv")
la_df = pd.read_csv("./stimulus_preprocessed/low_arousal_1_normalized.csv")
t_values = []
p_values = []
for column in columns:
    ha_hf = ha_df[column].values
    la_hf = la_df[column].values
    t_value, p_value = stats.ttest_ind(ha_hf, la_hf)
    t_values.append(t_value)
    p_values.append(p_value)
    
for i, p_value in enumerate(p_values):
    if p_value<0.05:
        print(f"p value of {columns[i]} is {p_value}")

# for i, p_value in enumerate(p_values):
#     print(f"{p_value:.3f}")