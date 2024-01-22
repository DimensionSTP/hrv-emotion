HYDRA_FULL_ERROR=1 python tune.py --multirun condition=arousal,valence tuning_way=original,cv
HYDRA_FULL_ERROR=1 python tune.py --config-name=stimulus_xgb_classifier_tune.yaml
HYDRA_FULL_ERROR=1 python tune.py --config-name=stimulus_xgb_classifier_tune.yaml condition=valence