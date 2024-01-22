HYDRA_FULL_ERROR=1 python test.py dataset=sl_rppg
HYDRA_FULL_ERROR=1 python test.py dataset=sl_rppg condition=valence
HYDRA_FULL_ERROR=1 python test.py --config-name=stimulus_xgb_classifier_test.yaml dataset=sl_rppg
HYDRA_FULL_ERROR=1 python test.py --config-name=stimulus_xgb_classifier_test.yaml dataset=sl_rppg condition=valence