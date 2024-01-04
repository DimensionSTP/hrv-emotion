import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import plot_importance
import matplotlib.pyplot as plt


def emotion_classifier(
    meaningful_features: list, 
    df_path: str, 
    result_path: str, 
    emotion: str, 
    standard: str, 
    condition: str, 
    filter_outlier: bool,
    classifier: str, 
    num_folds: int, 
    fold_seed: int,
    plt_save_path: str,
):
    df = pd.read_excel(df_path)
    
    # people_top = ['하현옥', '김두용', '이정한', '변남윤', '송영달', '이민정', '안만석', '정희수', '백지영', '최시리', '김주일', '이선진', '양정안', '김채연', '권윤경', '박유민', '정민경', '김태현', '김형민', '정승아', '이진성', '오명숙', '유은식', '김은미', '이점석']
    # people_bottom = ['신금숙', '박세익', '송효석', '배성우', '손하늘', '최태진', '윤호영', '김설양', '표성민', '김채윤', '최사랑', '최서영', '서정호', '김지연', '서창희', '김영재', '유현지', '김송이', '임우준', '박동수', '김세희', '주용현', '류채원', '홍유진', '김태양']
    # df = df[df["이름"].isin(people_bottom)]
    
    if emotion == "24":
        if standard == "compound":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6) & (df["자극_Arousal"] == 1) & (df["Valence"] <= 2) & (df["자극_Valence"] == 0)]
                low = df[(df["Arousal"] <= 2) & (df["자극_Arousal"] == 0) & (df["Valence"] >= 6) & (df["자극_Valence"] == 1)]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5) & (df["자극_Arousal"] == 1) & (df["Valence"] <= 3) & (df["자극_Valence"] == 0)]
                low = df[(df["Arousal"] <= 3) & (df["자극_Arousal"] == 0) & (df["Valence"] >= 5) & (df["자극_Valence"] == 1)]
            else:
                raise ValueError("Invalid condition")
        elif standard == "survey":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6) & (df["Valence"] <= 2)]
                low = df[(df["Arousal"] <= 2) & (df["Valence"] >= 6)]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5) & (df["Valence"] <= 3)]
                low = df[(df["Arousal"] <= 3) & (df["Valence"] >= 5)]
            else:
                raise ValueError("Invalid condition")
        else:
            raise ValueError("Invalid standard")
    elif emotion == "arousal":
        if standard == "compound":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6) & (df["자극_Arousal"] == 1)]
                low = df[(df["Arousal"] <= 2) & (df["자극_Arousal"] == 0)]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5) & (df["자극_Arousal"] == 1)]
                low = df[(df["Arousal"] <= 3) & (df["자극_Arousal"] == 0)]
            else:
                raise ValueError("Invalid condition")
        elif standard == "survey":
            if condition == "extreme":
                high = df[(df["Arousal"] >= 6)]
                low = df[(df["Arousal"] <= 2)]
            elif condition == "normal":
                high = df[(df["Arousal"] >= 5)]
                low = df[(df["Arousal"] <= 3)]
            else:
                raise ValueError("Invalid condition")
        else:
            raise ValueError("Invalid standard")
    elif emotion == "valence":
        if standard == "compound":
            if condition == "extreme":
                high = df[(df["Valence"] >= 6) & (df["자극_Valence"] == 1)]
                low = df[(df["Valence"] <= 2) & (df["자극_Valence"] == 0)]
            elif condition == "normal":
                high = df[(df["Valence"] >= 5) & (df["자극_Valence"] == 1)]
                low = df[(df["Valence"] <= 3) & (df["자극_Valence"] == 0)]
            else:
                raise ValueError("Invalid condition")
        elif standard == "survey":
            if condition == "extreme":
                high = df[(df["Valence"] >= 6)]
                low = df[(df["Valence"] <= 2)]
            elif condition == "normal":
                high = df[(df["Valence"] >= 5)]
                low = df[(df["Valence"] <= 3)]
            else:
                raise ValueError("Invalid condition")
        else:
            raise ValueError("Invalid standard")
    else:
        raise ValueError("Invalid emotion")
    
    if filter_outlier ==  True:
        is_outlier_filterd = "True"
        groups = [high, low]
        filtered_groups = []
        num_group_outliers = []
        for group in groups:
            group_mean_values = group.mean()
            group_std_values = group.std()
            group["outlier"] = ((group > group_mean_values + 3 * group_std_values) | (group < group_mean_values - 3 * group_std_values)).any(axis=1).astype(int)
            num_group_outlier = len(group.loc[group["outlier"] == 1])
            group = group[(group["outlier"] == 0)]
            filtered_groups.append(group)
            num_group_outliers.append(num_group_outlier)
        high = filtered_groups[0]
        low = filtered_groups[1]
    elif filter_outlier == False:
        is_outlier_filterd = "False"
        num_group_outliers = [0, 0]
        high = high
        low = low
    else:
        raise ValueError("Invalid filter_outlier")
    
    high["label"] = 1
    low["label"] = 0
    dataset = pd.concat([high, low], ignore_index=True)
    
    data = dataset[meaningful_features]
    label = dataset["label"]
    
    kf = KFold(n_splits=num_folds, random_state=fold_seed, shuffle=True)
    if classifier == "basic":
        clf = lgb.LGBMClassifier()
    
    accs = []
    for i, idx in enumerate(kf.split(data, label)):
        train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
        val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]
        
        clf.fit(train_data, train_label)
        
        pred = clf.predict(val_data)
        accuracy = accuracy_score(pred, val_label)
        print(f"{i + 1} Fold Accuracy = {accuracy}")
        accs.append(accuracy)
    avg_acc = np.mean(accs)
    print(avg_acc)
    
    avg_acc_percent = np.around(100 * avg_acc, 2)
    avg_acc_percent_for_name = int(np.around(100 * avg_acc))
    
    result = {
        "분류 감성" : emotion,
        "그룹 분류 기준" : standard,
        "주관 평가 경계" : condition,
        "사용된 HRV 지표" : meaningful_features,
        "이상치 제거 여부" : is_outlier_filterd, 
        "분류기 종류" : classifier,
        "Kfold 수" : num_folds,
        "평균 정확도(%)" : avg_acc_percent
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
    
    fig, ax = plt.subplots(figsize=(10,12))
    plot_importance(clf, ax=ax)
    # plt.show()
    if not os.path.exists(plt_save_path):
        os.makedirs(plt_save_path)
    plt.savefig(f"./{plt_save_path}/{emotion}_{standard}_{condition}_{classifier}_{num_folds}_{avg_acc_percent_for_name}.png")

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
        # "dHz",
        "pPow",
        # "pHz",
        "CohRatio",
        "RSA_PB",
    ]
        
    DF_PATH = "./tabular_dataset/survey_normalized_80.xlsx"
    RESULT_PATH = "./predict_results/classifier_results_80.csv"
    EMOTIONS = ["24", "arousal", "valence"]
    STANDARDS = ["compound", "survey"]
    CONDITIONS = ["extreme", "normal"]
    CLASSIFIER = "basic"
    NUM_FOLDS = 10
    FOLD_SEED = 2023
    PLT_SAVE_PATH = "classifier_reuslt_importance_figures/basic_solidly_normalized_80_augmented_features_all_not_vlf_hz"
    
    for emotion in EMOTIONS:
        for standard in STANDARDS:
            for condition in CONDITIONS:
                emotion_classifier(
                    meaningful_features=MEANINGFUL_FEATURES,
                    df_path=DF_PATH,
                    result_path=RESULT_PATH,
                    emotion=emotion,
                    standard=standard,
                    condition=condition,
                    filter_outlier=True,
                    classifier=CLASSIFIER,
                    num_folds=NUM_FOLDS,
                    fold_seed=FOLD_SEED,
                    plt_save_path=PLT_SAVE_PATH,
                )