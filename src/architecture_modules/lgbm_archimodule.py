import os
from typing import Union
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import lightgbm as lgb
from lightgbm import plot_importance

import wandb
from wandb.lightgbm import wandb_callback, log_summary

import matplotlib.pyplot as plt


class LGBMClassifierModule():
    def __init__(
        self,
        model_name: str,
        model_save_path: str,
        result_path: str,
    ) -> None:
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.result_path = result_path

    def train(
        self, 
        data: pd.DataFrame,
        label: pd.Series,
        num_folds: int, 
        seed: int,
        params_path: Union[str, bool],
        result_name: str,
        plt_save_path: str,
    ) -> None:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        if params_path:
            params = json.load(open(f"{params_path}/best_params.json", "rt", encoding="UTF-8"))
            params["verbose"] = -1
        else:
            params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "binary_logloss",
                "seed": seed,
            }

        wandb.init(project="hrv-emotion", entity="DimensionSTP", name=self.model_name)

        accs = []
        f1s = []
        for i, idx in enumerate(tqdm(kf.split(data, label))):
            train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
            val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]
            train_dataset = lgb.Dataset(train_data, train_label)
            val_dataset = lgb.Dataset(val_data, val_label)

            classifier = lgb.train(
                params,
                train_dataset,
                valid_sets=[train_dataset, val_dataset],
                valid_names=("validation"),
                callbacks=[wandb_callback()],
            )
            log_summary(classifier, save_model_checkpoint=True)

            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            classifier.save_model(f"{self.model_save_path}/fold{i}.txt")

            pred = classifier.predict(val_data)
            pred_binary = np.where(pred > 0.5, 1 , 0)
            accuracy = accuracy_score(val_label, pred_binary)
            f1 = f1_score(val_label, pred_binary)
            accs.append(accuracy)
            f1s.append(f1)
        avg_acc = np.mean(accs)
        avg_f1 = np.mean(f1s)
        print(f"average accuracy : {avg_acc}")
        print(f"average f1 score : {avg_f1}")

        avg_acc_percent = np.around(100 * avg_acc, 2)
        avg_acc_percent_for_name = int(np.around(100 * avg_acc))
        avg_f1_percent = np.around(100 * avg_f1, 2)
        avg_f1_percent_for_name = int(np.around(100 * avg_f1))

        result = {
            "분류기 종류": self.model_name,
            "사용된 지표": data.columns.tolist(),
            "Kfold 수": num_folds,
            "평균 정확도(%)": avg_acc_percent,
            "평균 f1(%)": avg_f1_percent,
        }
        result_df = pd.DataFrame.from_dict(result, orient="index").T

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        result_file = f"{self.result_path}/{result_name}"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat([original_result_df, result_df], ignore_index=True)
            new_result_df.to_csv(
            result_file, 
            encoding="utf-8-sig", 
            index=False,
            )
        else:
            result_df.to_csv(
            result_file, 
            encoding="utf-8-sig", 
            index=False,
            )

        fig, ax = plt.subplots(figsize=(10,12))
        plot_importance(classifier, ax=ax)
        if not os.path.exists(plt_save_path):
            os.makedirs(plt_save_path)
        plt.savefig(f"{plt_save_path}/{self.model_name}_{num_folds}_{avg_acc_percent_for_name}_{avg_f1_percent_for_name}.png")

    def test(
        self, 
        data: pd.DataFrame,
        label: pd.Series,
        result_name: str,
    ) -> None:
        pred_mean = np.zeros((len(data),))
        for model_file in (tqdm(os.listdir(self.model_save_path))):
            classifier = lgb.Booster(model_file=f"{self.model_save_path}/{model_file}")
            pred = classifier.predict(data) / len((os.listdir(self.model_save_path)))
            pred_mean += pred
        pred_binaries = np.around(pred_mean).astype(int)
        accuracy = accuracy_score(label, pred_binaries)
        f1 = f1_score(label, pred_binaries)
        print(f"average accuracy : {accuracy}")
        print(f"average f1 score : {f1}")
        acc_percent = np.around(100 * accuracy, 2)
        f1_percent = np.around(100 * f1, 2)

        result = {
            "분류기 종류": self.model_name,
            "사용된 지표": data.columns.tolist(),
            "평균 정확도(%)": acc_percent,
            "평균 f1(%)": f1_percent,
        }
        result_df = pd.DataFrame.from_dict(result, orient="index").T

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        result_file = f"{self.result_path}/{result_name}"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat([original_result_df, result_df], ignore_index=True)
            new_result_df.to_csv(
            result_file, 
            encoding="utf-8-sig", 
            index=False,
            )
        else:
            result_df.to_csv(
            result_file, 
            encoding="utf-8-sig", 
            index=False,
            )