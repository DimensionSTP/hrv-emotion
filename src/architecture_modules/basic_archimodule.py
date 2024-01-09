import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

import lightgbm as lgb
from lightgbm import plot_importance

import wandb
from wandb.lightgbm import wandb_callback, log_summary

import matplotlib.pyplot as plt


class BasicClassifierModule():
    def __init__(
        self,
        condition: str,
        model_save_path: str,
        result_path: str,
    ):
        self.condition = condition
        self.model_save_path = model_save_path
        self.result_path = result_path

    def train(
        self, 
        data: pd.DataFrame,
        label: pd.Series,
        num_folds: int, 
        fold_seed: int,
        boosting_type: str,
        objective: str,
        metric: str,
        result_name: str,
        plt_save_path: str,
    ) -> None:
        kf = KFold(n_splits=num_folds, random_state=fold_seed, shuffle=True)
        params = dict()
        params["boosting_type"] = boosting_type
        params["objective"] = objective
        params["metric"] = metric
        wandb.init(project="hrv-emotion", entity="DimensionSTP", name="basic")

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
                num_boost_round=100,
                valid_sets=[train_dataset, val_dataset],
                valid_names=("validation"),
                callbacks=[wandb_callback()],
            )
            log_summary(classifier, save_model_checkpoint=True)

            if not os.path.exists(f"{self.model_save_path}/{self.condition}"):
                os.makedirs(f"{self.model_save_path}/{self.condition}")
            classifier.save_model(f"{self.model_save_path}/{self.condition}/fold{i}.txt")

            pred = classifier.predict(val_data)
            pred_binary = np.where(pred > 0.5, 1 , 0)
            accuracy = accuracy_score(pred_binary, val_label)
            f1 = f1_score(pred_binary, val_label)
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
            "분류 감성": self.condition,
            "사용된 HRV 지표": data.columns.tolist(),
            "분류기 종류": "basic",
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
        plt.savefig(f"{plt_save_path}/{self.condition}_basic_{num_folds}_{avg_acc_percent_for_name}_{avg_f1_percent_for_name}.png")

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
        accuracy = accuracy_score(pred_binaries, label)
        f1 = f1_score(pred_binaries, label)
        print(f"average accuracy : {accuracy}")
        print(f"average f1 score : {f1}")
        acc_percent = np.around(100 * accuracy, 2)
        f1_percent = np.around(100 * f1, 2)

        result = {
            "분류 감성": self.condition,
            "사용된 HRV 지표": data.columns.tolist(),
            "분류기 종류": "basic",
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