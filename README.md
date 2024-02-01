# HRV-emotion package

## HRV-emotion package including machine learning and statistical analysis

### Dataset
Lab datasets collected from experiments

### 🚀Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/hrv-emotion.git
cd hrv-emotion

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

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

* concat augmented dataset for machine learning train dataset
```shell
python concat_augmented_dataset.py
```

### Machine Learning(HRV-emotion classifier)

* model tuning
```shell
python main.py mode=tune condition={arousal or valence}
```

* model training
```shell
python main.py mode=train condition={arousal or valence} is_tuned={bool}
```

* model testing
```shell
python main.py mode=test condition={arousal or valence} data_type={test data}
```

* all model tuning
```shell
./scripts/tune.sh
```

* all model training
```shell
./scripts/train.sh
```

* all model testing
```shell
./scripts/test.sh
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__