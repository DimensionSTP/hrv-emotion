import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.multicomp import MultiComparison

columns = [
    "RRI",
    "BPM",
    "SDNN",
    "rMSSD",
    "VLF",
    "LF",
    "HF",
    "HFp",
    "lnVLF",
    "lnLF",
    "lnHF",
    "LF/HF",
    "tPow",
    "dPow",
    "pPow",
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

arousal["group"] = "arousal"
relaxation["group"] = "relaxation"
neutral["group"] = "neutral"

for column in columns:
    data = pd.concat([arousal, relaxation, neutral], ignore_index=True)
    comp = MultiComparison(data[column], data["group"])
    result = comp.allpairtest(stats.ttest_ind, method='bonf')
    print(f"feature : {column}")
    print(result[0])