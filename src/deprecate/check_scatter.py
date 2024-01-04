import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel("survey_normalized_80.xlsx")

survey_scores = ["Arousal", "Valence"]
features = [
    "RRI",
    "BPM",
    "SDNN",
    "rMSSD",
    # "pNN50",
    # "VLF",
    "LF",
    "HF",
    # "VLFp",
    # "LFp",
    # "HFp",
    # "lnVLF",
    "lnLF",
    "lnHF",
    # "VLF/HF",
    # "LF/HF",
    "tPow",
    "dPow",
    "dHz",
    "pPow",
    "pHz",
    "CohRatio",
    "RSA_PB",
]

for survey_score in survey_scores:
    for feature in features:
        x = df[survey_score]
        y = df[feature]

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y)
        plt.xlabel(survey_score)
        plt.ylabel(feature)
        plt.grid(True)
        plt.savefig(f"./survey-feature_scatter/{survey_score}/{survey_score}_{feature}.png")