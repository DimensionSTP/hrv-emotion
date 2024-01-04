import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_excel("survey.xlsx")
df["lnHF/lnLF"] = np.nan
df["lnHF/lnLF"] = df["lnHF"] / df["lnLF"]

df["ln(HF/LF)"] = np.nan
df["ln(HF/LF)"] = np.log(df["HF"] / df["LF"])
df["stimuli_arousal"] = np.nan
df.loc[df["Arousal"] >= 5, "stimuli_arousal"] = "각성"
df.loc[df["Arousal"] == 4, "stimuli_arousal"] = "중립"
df.loc[df["Arousal"] <= 3, "stimuli_arousal"] = "이완"

df_arousal = df[df["stimuli_arousal"] == "각성"]
df_neutral = df[df["stimuli_arousal"] == "중립"]
df_relax = df[df["stimuli_arousal"] == "이완"]

x_arousal = df_arousal["lnHF"].values
y_arousal = df_arousal["ln(HF/LF)"].values
x_arousal = x_arousal.reshape(-1, 1)
y_arousal = y_arousal.reshape(-1, 1)

x_neutral = df_neutral["lnHF"].values
y_neutral = df_neutral["ln(HF/LF)"].values
x_neutral = x_neutral.reshape(-1, 1)
y_neutral = y_neutral.reshape(-1, 1)

x_relax = df_relax["lnHF"].values
y_relax = df_relax["ln(HF/LF)"].values
x_relax = x_relax.reshape(-1, 1)
y_relax = y_relax.reshape(-1, 1)

lr_arousal = LinearRegression()
lr_neutral = LinearRegression()
lr_relax = LinearRegression()

lr_arousal.fit(x_arousal, y_arousal)
lr_neutral.fit(x_neutral, y_neutral)
lr_relax.fit(x_relax, y_relax)

results_arousal = sm.OLS(y_arousal, sm.add_constant(x_arousal)).fit()
results_neutral = sm.OLS(y_neutral, sm.add_constant(x_neutral)).fit()
results_relax = sm.OLS(y_relax, sm.add_constant(x_relax)).fit()

print(lr_arousal.coef_[0])
print(lr_arousal.intercept_)
print(lr_neutral.coef_[0])
print(lr_neutral.intercept_)
print(lr_relax.coef_[0])
print(lr_relax.intercept_)

print(results_arousal.summary())
print(results_neutral.summary())
print(results_relax.summary())