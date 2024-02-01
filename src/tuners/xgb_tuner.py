import os
from typing import Dict, Any, Tuple
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


class XGBTuner():
    def __init__(
        self,
        hparams: Dict[str, Any],
        data: pd.DataFrame,
        label: pd.Series,
        split_size: float,
        num_folds: int,
        metric: str,
        num_trials: int,
        seed: int,
        tuning_way: str,
        hparams_save_path: str,
    ) -> None:
        self.hparams = hparams
        self.data = data
        self.label = label
        self.split_size = split_size
        self.num_folds = num_folds
        self.metric = metric
        self.num_trials = num_trials
        self.seed = seed
        self.tuning_way = tuning_way
        self.hparams_save_path = hparams_save_path

    def __call__(self) -> None:
        if self.tuning_way == "original":
            study=optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.seed), pruner=HyperbandPruner())
            study.optimize(self.optuna_objective, n_trials=self.num_trials)
            trial = study.best_trial
            best_score = trial.value
            best_params = trial.params
        else:
            raise ValueError("Invalid tuning way")

        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(self.hparams_save_path, exist_ok=True)

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(best_params, json_file)

    def get_split_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        train_data, val_data, train_label, val_label = train_test_split(
            self.data, 
            self.label, 
            test_size=self.split_size,
            random_state=self.seed,
            shuffle=True,
        )
        return (train_data, val_data, train_label, val_label)

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        params = dict()
        if self.hparams.objective:
            params["objective"] = self.hparams.objective
        if self.hparams.eval_metric:
            params["eval_metric"] = self.hparams.eval_metric
        if self.hparams.booster:
            params["booster"] = trial.suggest_categorical(
                name="booster",
                choices=self.hparams.booster,
            )
        if self.hparams.lambda_:
            params["lambda"] = trial.suggest_loguniform(
                name="lambda",
                low=self.hparams.lambda_.low,
                high=self.hparams.lambda_.high,
            )
        if self.hparams.alpha:
            params["alpha"] = trial.suggest_loguniform(
                name="alpha",
                low=self.hparams.alpha.low,
                high=self.hparams.alpha.high,
            )
        if self.hparams.max_depth:
            params["max_depth"] = trial.suggest_int(
                name="max_depth",
                low=self.hparams.max_depth.low,
                high=self.hparams.max_depth.high,
                log=self.hparams.max_depth.log,
            )
        if self.hparams.eta:
            params["eta"] = trial.suggest_loguniform(
                name="eta",
                low=self.hparams.eta.low,
                high=self.hparams.eta.high,
            )
        if self.hparams.gamma:
            params["gamma"] = trial.suggest_loguniform(
                name="gamma",
                low=self.hparams.gamma.low,
                high=self.hparams.gamma.high,
            )
        if self.hparams.subsample:
            params["subsample"] = trial.suggest_uniform(
                name="subsample",
                low=self.hparams.subsample.low,
                high=self.hparams.subsample.high,
            )
        if self.hparams.colsample_bytree:
            params["colsample_bytree"] = trial.suggest_uniform(
                name="colsample_bytree",
                low=self.hparams.colsample_bytree.low,
                high=self.hparams.colsample_bytree.high,
            )

        train_data, val_data, train_label, val_label = self.get_split_dataset()

        classifier = xgb.XGBClassifier(**params, random_state=self.seed)
        classifier.fit(train_data, train_label)
        pred = classifier.predict(val_data)

        if self.metric == "accuracy":
            score = accuracy_score(val_label, pred)
        elif self.metric == "f1":
            score = f1_score(val_label, pred)
        else:
            raise ValueError("Invalid metric")
        return score