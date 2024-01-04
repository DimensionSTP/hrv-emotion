import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb


def emotion_classifier(
    meaningful_features: list,
    df_path: str,
    test_df_path: str,
    result_path: str,
    condition: str,
    classifier: str,
    num_folds: int,
    fold_seed: int,
    model_save_path: str,
):
    df = pd.read_excel(df_path)
    test_df = pd.read_excel(test_df_path)

    data = df[meaningful_features]
    test_data = test_df[meaningful_features]
    if condition == "arousal":
        label = df["자극_Arousal"]
        test_df["arousal_predict"] = 0
    elif condition == "valence":
        label = df["자극_Valence"]
        test_df["valence_predict"] = 0
    elif condition == "dimensional":
        df["label"] = 0
        df["label"] = df.apply(
            lambda row: 1
            if row["자극_Arousal"] == 1 and row["자극_Valence"] == 1
            else 2
            if row["자극_Arousal"] == 1 and row["자극_Valence"] == 0
            else 3
            if row["자극_Arousal"] == 0 and row["자극_Valence"] == 0
            else 4,
            axis=1,
        )
        label = df["label"]
        test_df["dimensional_predict"] = 0
    else:
        raise ValueError("Invalid condition")

    kf = KFold(n_splits=num_folds, random_state=fold_seed, shuffle=True)
    if classifier == "basic":
        clf = lgb.LGBMClassifier()

    pred_all_mean = np.zeros((len(test_df),))
    accs = []
    for i, idx in enumerate(kf.split(data, label)):
        train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
        val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]

        clf.fit(train_data, train_label)

        if not os.path.exists(f"{model_save_path}/{condition}"):
            os.makedirs(f"{model_save_path}/{condition}")
        clf.booster_.save_model(f"{model_save_path}/{condition}/fold{i}.txt")

        pred = clf.predict(val_data)
        pred_all = clf.predict(test_data) / kf.n_splits
        pred_all_mean += pred_all
        accuracy = accuracy_score(pred, val_label)
        print(f"{i + 1} Fold Accuracy = {accuracy}")
        accs.append(accuracy)

    avg_acc = np.mean(accs)
    print(avg_acc)
    avg_acc_percent = np.around(100 * avg_acc, 2)
    pred_all_result = np.around(pred_all_mean).astype(int)
    test_df.iloc[:, -1] = pred_all_result

    result = {
        "주관 평가 경계": condition,
        "사용된 HRV 지표": meaningful_features,
        "분류기 종류": classifier,
        "Kfold 수": num_folds,
        "평균 정확도(%)": avg_acc_percent,
    }
    result_df = pd.DataFrame.from_dict(result, orient="index").T

    if os.path.isfile(result_path):
        original_result_df = pd.read_csv(result_path)
        new_result_df = pd.concat([original_result_df, result_df], ignore_index=True)
        new_result_df.to_csv(
            result_path,
            encoding="utf-8-sig",
            index=False,
        )
    else:
        result_df.to_csv(
            result_path,
            encoding="utf-8-sig",
            index=False,
        )

    test_df.to_excel(
        test_df_path,
        sheet_name="normalized",
        index=False,
    )


if __name__ == "__main__":
    MEANINGFUL_FEATURES = [
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
    DF_PATH = "./tabular_dataset/survey_solidly_normalized_180_total_augmented.xlsx"
    TEST_DF_PATH = (
        "./tabular_dataset/survey_solidly_normalized_rppg_180_all_features_trained.xlsx"
    )
    RESULT_PATH = "./predict_results/stimulus_classifier_solidly_normalized_180_augmented_rppg_result.csv"
    CONDITIONS = ["arousal", "valence", "dimensional"]
    CLASSIFIER = "basic"
    NUM_FOLDS = 10
    FOLD_SEED = 2023
    MODEL_SAVE_PATH = (
        "./save_model/basic_classifier/solidly_normalized_all_features_rppg_180"
    )

    for condition in CONDITIONS:
        emotion_classifier(
            meaningful_features=MEANINGFUL_FEATURES,
            df_path=DF_PATH,
            test_df_path=TEST_DF_PATH,
            result_path=RESULT_PATH,
            condition=condition,
            classifier=CLASSIFIER,
            num_folds=NUM_FOLDS,
            fold_seed=FOLD_SEED,
            model_save_path=MODEL_SAVE_PATH,
        )
