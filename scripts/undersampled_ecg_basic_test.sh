HYDRA_FULL_ERROR=1 python test.py --multirun dataset=sl_50,sl_100,sl_250 condition=arousal,valence
HYDRA_FULL_ERROR=1 python test.py --multirun --config-name=stimulus_xgb_classifier_test.yaml dataset=sl_50,sl_100,sl_250 condition=arousal,valence