import os
from typing import Dict, Any, Tuple
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

import optuna
from optuna.integration.lightgbm import LightGBMTunerCV
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


class LGBMTuner():
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
        elif self.tuning_way == "cv":
            tuner = self.get_optuna_cv_tuner()
            tuner.run()
            best_score = tuner.best_score
            best_params = tuner.best_params
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
        params["seed"] = self.seed
        if self.hparams.objective:
            params["objective"] = self.hparams.objective
        if self.hparams.metric:
            params["metric"] = self.hparams.metric
        if self.hparams.verbosity:
            params["verbosity"] = self.hparams.verbosity
        if self.hparams.boosting_type:
            params["boosting_type"] = self.hparams.boosting_type
        if self.hparams.learning_rate:
            params["learning_rate"] = trial.suggest_float(
                name="learning_rate",
                low=self.hparams.learning_rate.low,
                high=self.hparams.learning_rate.high,
                log=self.hparams.learning_rate.log,
            )
        if self.hparams.n_estimators:
            params["n_estimators"] = trial.suggest_int(
                name="n_estimators",
                low=self.hparams.n_estimators.low,
                high=self.hparams.n_estimators.high,
                log=self.hparams.n_estimators.log,
            )
        if self.hparams.lambda_l1:
            params["lambda_l1"] = trial.suggest_loguniform(
                name="lambda_l1",
                low=self.hparams.lambda_l1.low,
                high=self.hparams.lambda_l1.high,
            )
        if self.hparams.lambda_l2:
            params["lambda_l2"] = trial.suggest_loguniform(
                name="lambda_l2",
                low=self.hparams.lambda_l2.low,
                high=self.hparams.lambda_l2.high,
            )
        if self.hparams.num_leaves:
            params["num_leaves"] = trial.suggest_int(
                name="num_leaves",
                low=self.hparams.num_leaves.low,
                high=self.hparams.num_leaves.high,
                log=self.hparams.num_leaves.log,
            )
        if self.hparams.max_depth:
            params["max_depth"] = trial.suggest_int(
                name="max_depth",
                low=self.hparams.max_depth.low,
                high=self.hparams.max_depth.high,
                log=self.hparams.max_depth.log,
            )
        if self.hparams.feature_fraction:
            params["feature_fraction"] = trial.suggest_uniform(
                name="feature_fraction",
                low=self.hparams.feature_fraction.low,
                high=self.hparams.feature_fraction.high,
            )
        if self.hparams.bagging_fraction:
            params["bagging_fraction"] = trial.suggest_uniform(
                name="bagging_fraction",
                low=self.hparams.bagging_fraction.low,
                high=self.hparams.bagging_fraction.high,
            )
        if self.hparams.bagging_freq:
            params["bagging_freq"] = trial.suggest_int(
                name="bagging_freq",
                low=self.hparams.bagging_freq.low,
                high=self.hparams.bagging_freq.high,
                log=self.hparams.bagging_freq.log,
            )
        if self.hparams.min_child_samples:
            params["min_child_samples"] = trial.suggest_int(
                name="min_child_samples",
                low=self.hparams.min_child_samples.low,
                high=self.hparams.min_child_samples.high,
                log=self.hparams.min_child_samples.log,
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
        if self.hparams.reg_alpha:
            params["reg_alpha"] = trial.suggest_uniform(
                name="reg_alpha",
                low=self.hparams.reg_alpha.low,
                high=self.hparams.reg_alpha.high,
            )
        if self.hparams.reg_lambda:
            params["reg_lambda"] = trial.suggest_uniform(
                name="reg_lambda",
                low=self.hparams.reg_lambda.low,
                high=self.hparams.reg_lambda.high,
            )

        train_data, val_data, train_label, val_label = self.get_split_dataset()
        train_dataset = lgb.Dataset(train_data, train_label)
        val_dataset = lgb.Dataset(val_data, val_label)

        classifier = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=("validation"),
        )
        pred = classifier.predict(val_data)
        pred_binary = np.where(pred > 0.5, 1 , 0)

        if self.metric == "accuracy":
            score = accuracy_score(val_label, pred_binary)
        elif self.metric == "f1":
            score = f1_score(val_label, pred_binary)
        else:
            raise ValueError("Invalid metric")
        return score

    def get_optuna_cv_tuner(self) -> LightGBMTunerCV:
        params = dict()
        params["seed"] = self.seed
        if self.hparams.objective:
            params["objective"] = self.hparams.objective
        if self.hparams.metric:
            params["metric"] = self.hparams.metric
        if self.hparams.verbosity:
            params["verbosity"] = self.hparams.verbosity
        if self.hparams.boosting_type:
            params["boosting_type"] = self.hparams.boosting_type
        
        train_dataset = lgb.Dataset(self.data, self.label)

        tuner = LightGBMTunerCV(
            params,
            train_set=train_dataset,
            folds=StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed),
            callbacks=[early_stopping(self.num_trials), log_evaluation(self.num_trials)],
        )
        return tuner