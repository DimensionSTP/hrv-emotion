HYDRA_FULL_ERROR=1 python main.py --multirun mode=test data_type=sl_test_dataset condition=arousal,valence
HYDRA_FULL_ERROR=1 python main.py --multirun --config-name=xgb.yaml mode=test data_type=sl_test_dataset condition=arousal,valence