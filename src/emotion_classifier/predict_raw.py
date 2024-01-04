import os

import numpy as np
import pandas as pd
import lightgbm as lgb


def hrv_emotion_predict(
    meaningful_features: list, 
    test_df_path: str, 
    condition: str, 
    model_path: str, 
):
    test_df = pd.read_excel(test_df_path)
    test_data = test_df[meaningful_features]
    
    model_files = os.listdir(f"{model_path}/{condition}")
    for i, model in enumerate(model_files):
        clf = lgb.Booster(model_file=f"{model_path}/{condition}/{model}")
        pred_all = clf.predict(test_data)
        test_df[f"{condition}_fold_{i}_predict"] = pred_all
    test_df.to_excel(
        test_df_path, 
        sheet_name="solidly_normalized",
        index=False,
    )


if __name__ == "__main__":
    MEANINGFUL_FEATURES = [
        "RRI",
        "BPM",
        "SDNN",
        "rMSSD",
        # "pNN50",
        # "VLF",
        "LF",
        "HF",
        # "VLFp",
        "LFp",
        "HFp",
        # "lnVLF",
        "lnLF",
        "lnHF",
        # "VLF/HF",
        "LF/HF",
        "tPow",
        "dPow",
        "dHz",
        "pPow",
        "pHz",
        "CohRatio",
        "RSA_PB",
    ]
    TEST_DF_PATH = "./tabular_dataset/survey_solidly_normalized.xlsx"
    CONDITIONS = ["arousal", "valence"]
    MODEL_PATH = "./save_model/basic_classifier/solidly_normalized_all_features"
    
    for condition in CONDITIONS:
        hrv_emotion_predict(
            meaningful_features=MEANINGFUL_FEATURES,
            test_df_path=TEST_DF_PATH,
            condition=condition,
            model_path = MODEL_PATH
        )