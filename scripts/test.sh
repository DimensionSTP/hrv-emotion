HYDRA_FULL_ERROR=1 python test.py
HYDRA_FULL_ERROR=1 python test.py condition=valence
HYDRA_FULL_ERROR=1 python test.py --config-name=stimulus_xgb_classifier_test.yaml
HYDRA_FULL_ERROR=1 python test.py --config-name=stimulus_xgb_classifier_test.yaml condition=valence