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
    pred_all_mean = np.zeros((len(test_df),))
    for model in model_files:
        clf = lgb.Booster(model_file=f"{model_path}/{condition}/{model}")
        if condition == "dimensional":
            pred = clf.predict(test_data)
            pred_all = (np.argmax(pred, axis=1) + 1) / len(model_files)
        else:
            pred_all = clf.predict(test_data) / len(model_files)
        pred_all_mean += pred_all
        
    pred_all_result = np.around(pred_all_mean).astype(int)


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
    TEST_DF_PATH = "./tabular_dataset/survey_solidly_normalized_rppg_total_wo_vlfs_trained.xlsx"
    CONDITIONS = ["arousal", "valence", "dimensional"]
    MODEL_PATH = "./save_model/basic_classifier/solidly_normalized_wo_vlfs_rppg_130_total"
    
    for condition in CONDITIONS:
        hrv_emotion_predict(
            meaningful_features=MEANINGFUL_FEATURES,
            test_df_path=TEST_DF_PATH,
            condition=condition,
            model_path = MODEL_PATH
        )