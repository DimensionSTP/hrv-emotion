HYDRA_FULL_ERROR=1 python train.py
HYDRA_FULL_ERROR=1 python train.py condition=valence
HYDRA_FULL_ERROR=1 python train.py --config-name=stimulus_xgb_classifier_train.yaml
HYDRA_FULL_ERROR=1 python train.py --config-name=stimulus_xgb_classifier_train.yaml condition=valence