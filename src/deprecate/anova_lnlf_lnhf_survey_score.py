import pandas as pd
from scipy import stats

df = pd.read_excel("survey_normalized.xlsx")
arousal = df[(df["lnLF"]>=0) & (df["lnHF"]>=0)]
relaxation = df[(df["lnLF"]<0) & (df["lnHF"]<0)]
neutral1 = df[(df["lnLF"]>=0) & (df["lnHF"]<0)]
neutral2 = df[(df["lnLF"]<0) & (df["lnHF"]>=0)]
neutral =  pd.concat([neutral1, neutral2], ignore_index=True)

arousal_values = arousal["Valence"].values
relaxation_values = relaxation["Valence"].values
neutral_values = neutral["Valence"].values
f_value, p_value = stats.f_oneway(arousal_values, relaxation_values, neutral_values)

print(f"p value is {p_value}")