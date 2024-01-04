import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
# from catboost import CatBoostClassifier

# MEANINGFUL_FEATURES = ["SDNN", "rMSSD", "LF", "lnLF", "lnHF", "RRI", "BPM"]
# MEANINGFUL_FEATURES = ["SDNN", "rMSSD", "LF", "lnLF", "lnHF"]
MEANINGFUL_FEATURES = ["SDNN", "rMSSD","RRI", "BPM"]
# MEANINGFUL_FEATURES = ["SDNN", "rMSSD"]

# MEANINGFUL_FEATURES = ["lnLF", "lnHF"]
# MEANINGFUL_FEATURES = ["rMSSD", "HF", "lnLF", "lnHF", "RSA_PB"]
# MEANINGFUL_FEATURES = ["SDNN", "rMSSD", "LF", "HF", "lnLF", "lnHF", "tPow", "RSA_PB"]
# MEANINGFUL_FEATURES = ["RRI", "BPM", "SDNN", "rMSSD", "LF", "lnLF", "dPow", "pPow"]
# MEANINGFUL_FEATURES = ["RRI", "BPM", "SDNN", "rMSSD", "LF", "lnLF", "tPow", "dPow", "pPow"]

df = pd.read_excel("survey_normalized.xlsx")

# halv = df[(df["Arousal"] >= 6) & (df["자극_Arousal"]==1) & (df["Valence"] <= 2) & (df["자극_Valence"]==0)]
# lahv = df[(df["Arousal"] <= 2) & (df["자극_Arousal"]==0) & (df["Valence"] >= 6) & (df["자극_Valence"]==1)]

# halv = df[(df["Arousal"] >= 5) & (df["자극_Arousal"]==1) & (df["Valence"] <= 3) & (df["자극_Valence"]==0)]
# lahv = df[(df["Arousal"] <= 3) & (df["자극_Arousal"]==0) & (df["Valence"] >= 5) & (df["자극_Valence"]==1)]

# halv = df[(df["Arousal"] >= 6) & (df["Valence"] <= 2)]
# lahv = df[(df["Arousal"] <= 2) & (df["Valence"] >= 6)]

halv = df[(df["Arousal"] >= 5) & (df["Valence"] <= 3)]
lahv = df[(df["Arousal"] <= 3) & (df["Valence"] >= 5)]

halv["label"] = 1
lahv["label"] = 0
dataset = pd.concat([halv, lahv], ignore_index=True)


# ha = df[(df["Arousal"] >= 6) & (df["자극_Arousal"]==1)]
# la = df[(df["Arousal"] <= 2) & (df["자극_Arousal"]==0)]

# ha = df[(df["Arousal"] >= 5) & (df["자극_Arousal"]==1)]
# la = df[(df["Arousal"] <= 3) & (df["자극_Arousal"]==0)]

# ha = df[(df["Arousal"] >= 6)]
# la = df[(df["Arousal"] <= 2)

# ha = df[(df["Arousal"] >= 5)]
# la = df[(df["Arousal"] <= 3)]

# ha["label"] = 1
# la["label"] = 0
# dataset = pd.concat([ha, la], ignore_index=True)


# hv = df[(df["Valence"] >= 6) & (df["자극_Valence"]==1)]
# lv = df[(df["Valence"] <= 2) & (df["자극_Valence"]==0)]

# hv = df[(df["Valence"] >= 5) & (df["자극_Valence"]==1)]
# lv = df[(df["Valence"] <= 3) & (df["자극_Valence"]==0)]

# hv = df[(df["Valence"] >= 6)]
# lv = df[(df["Valence"] <= 2)]

# hv = df[(df["Valence"] >= 5)]
# lv = df[(df["Valence"] <= 3)]

# hv["label"] = 1
# lv["label"] = 0
# dataset = pd.concat([hv, lv], ignore_index=True)


# df = pd.read_csv("./stimulus_for_classifier/arousal_labeled.csv")
data = dataset[MEANINGFUL_FEATURES]
label = dataset["label"]

# data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.1, random_state=2023)
kf = KFold(n_splits=10, random_state=2023, shuffle=True)

clf = lgb.LGBMClassifier()
# clf = CatBoostClassifier(
#     iterations=9,
#     random_seed=42,
#     learning_rate=0.1,
#     custom_loss=['AUC', 'Accuracy']
# )

# clf.fit(data_train, label_train)
# pred = clf.predict(data_test)
# accuracy = accuracy_score(pred, label_test)
# print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(pred, label_test)))

accs = []
for i, idx in enumerate(kf.split(data, label)):
    train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
    val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]
    
    clf.fit(train_data, train_label)
    
    pred = clf.predict(val_data)
    accuracy = accuracy_score(pred, val_label)
    print(f"{i + 1} Fold Accuracy = {accuracy}")
    accs.append(accuracy)
print(np.mean(accs))