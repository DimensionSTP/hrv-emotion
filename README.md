# HRV-emotion package

## HRV-emotion package including machine learning and statistical analysis

### Dataset
Lab datasets collected from experiments

### ECG or rPPG signal to HRV

* rename ECG data files before preprocessing
```shell
python rename_thirty_dataset_files_in_directory.py
```

* rename rPPG data files before preprocessing
```shell
python rename_sl_rppg_dataset_files_in_directory.py
```

* signal to HRV
```shell
./scripts/signal_to_hrv.sh
```

### Preprocess HRV

* file path setting, normalization, averaging, make test dataset, statistical_analysis
```shell
./scripts/preprocess_hrv.sh
```

### Machine Learning(HRV-emotion classifier)

* basic LGBM classifier training
```shell
./scripts/basic_train.sh
```

* basic LGBM classifier testing(ECG)
```shell
./scripts/ecg_basic_test.sh
```

* basic LGBM classifier testing(rPPG)
```shell
./scripts/rppg_basic_test.sh
```