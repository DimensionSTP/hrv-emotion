import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# MEANINGFUL_FEATURES = ["rMSSD", "HF", "lnLF", "lnHF", "RSA_PB"]
# MEANINGFUL_FEATURES = ["RRI", "BPM", "SDNN", "rMSSD", "LF", "lnLF", "dPow", "pPow"]
# MEANINGFUL_FEATURES = ["RRI", "BPM", "SDNN", "rMSSD", "LF", "lnLF", "tPow", "dPow", "pPow"]
# MEANINGFUL_FEATURES = ["RRI", "BPM", "SDNN", "rMSSD", "LF", "lnLF", "lnHF", "tPow"]
MEANINGFUL_FEATURES = ["new_tPow"]
# MEANINGFUL_FEATURES = [
#     "RRI",
#     "BPM",
#     "SDNN",
#     "rMSSD",
#     # "pNN50",
#     "VLF",
#     "LF",
#     "HF",
#     "VLFp",
#     "LFp",
#     "HFp",
#     "lnVLF",
#     "lnLF",
#     "lnHF",
#     "VLF/HF",
#     "LF/HF",
#     "tPow",
#     "dPow",
#     "dHz",
#     "pPow",
#     "pHz",
#     "CohRatio",
#     "RSA_PB",
# ]

# df_high = pd.read_csv("survey_normalized_high_arousal_extreme.csv")
# df_low = pd.read_csv("survey_normalized_low_arousal_extreme.csv",)

# df_original = pd.read_excel("survey_normalized.xlsx")
# df_high = df_original[(df_original["Arousal"] >= 6) & (df_original["자극_Arousal"]==1) & (df_original["Valence"] <= 2) & (df_original["자극_Valence"]==0)]
# df_low = df_original[(df_original["Arousal"] <= 2) & (df_original["자극_Arousal"]==0) & (df_original["Valence"] >= 6) & (df_original["자극_Valence"]==1)]

df_original = pd.read_csv("./stmulus_preprocessed/reference_1_raw.csv")
df_original["new_tPow"] = df_original["tPow"] - df_original["VLF"]
names = ['권윤경', '김두용', '김설양', '김세희', '김송이', '김영재', '김은미', '김주일', '김지연', '김채연', '김채윤', '김태양', '김태현', '김형민', '류채원', '박동수', '박세익', '박유민', '배성우', '백지영', '변남윤', '서정호', '서창희', '손하늘', '송영달', '송효석', '신금숙', '안만석', '양정안', '오명숙', '유은식', '유현지', '윤호영', '이민정', '이선진', '이점석', '이정한', '이진성', '임우준', '정민경', '정승아', '정희수', '주용현', '최사랑', '최서영', '최시리', '최태진', '표성민', '하현옥', '홍유진']
df_original["이름"] = names
people_top = ['하현옥', '김두용', '이정한', '변남윤', '송영달', '이민정', '안만석', '정희수', '백지영', '최시리', '김주일', '이선진', '양정안', '김채연', '권윤경', '박유민', '정민경', '김태현', '김형민', '정승아', '이진성', '오명숙', '유은식', '김은미', '이점석']
people_bottom = ['신금숙', '박세익', '송효석', '배성우', '손하늘', '최태진', '윤호영', '김설양', '표성민', '김채윤', '최사랑', '최서영', '서정호', '김지연', '서창희', '김영재', '유현지', '김송이', '임우준', '박동수', '김세희', '주용현', '류채원', '홍유진', '김태양']
df_high = df_original[df_original["이름"].isin(people_top)]
df_low = df_original[df_original["이름"].isin(people_bottom)]

for feature in MEANINGFUL_FEATURES:
    df = pd.concat([df_high[feature], df_low[feature]], axis=1)
    # df.columns = ["HALV", "LAHV"]
    df.columns = ["High", "Low"]
    df_melted = pd.melt(df)
    df_melted.columns = [feature, "value"]
    sns.boxplot(data=df_melted, x=feature, y="value")
    plt.show()