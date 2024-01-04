import pandas as pd

MEANINGFUL_FEATURES = ["lnLF", "lnHF"]

def get_hrv_values(load_path, meaningful_features, save_path):
    hrv_data = pd.read_csv(load_path)
    hrv_data = hrv_data[meaningful_features]
    arousal = hrv_data[(hrv_data["lnLF"]>=0) & (hrv_data["lnHF"]>=0)]
    relaxation = hrv_data[(hrv_data["lnLF"]<0) & (hrv_data["lnHF"]<0)]
    neutral1 = hrv_data[(hrv_data["lnLF"]>=0) & (hrv_data["lnHF"]<0)]
    neutral2 = hrv_data[(hrv_data["lnLF"]<0) & (hrv_data["lnHF"]>=0)]
    neutral =  pd.concat([neutral1, neutral2], ignore_index=True)
    arousal_values = arousal.describe()
    relaxation_values = relaxation.describe()
    neutral1_values = neutral1.describe()
    neutral2_values = neutral2.describe()
    neutral_values = neutral.describe()
    arousal_values.to_csv(
        f"{save_path}_arousal.csv", 
        encoding="utf-8-sig",
        index=True
        )
    relaxation_values.to_csv(
        f"{save_path}_relaxation.csv", 
        encoding="utf-8-sig",
        index=True
        )
    neutral1_values.to_csv(
        f"{save_path}_neutral1.csv", 
        encoding="utf-8-sig",
        index=True
        )
    neutral2_values.to_csv(
        f"{save_path}_neutral2.csv", 
        encoding="utf-8-sig",
        index=True
        )
    neutral_values.to_csv(
        f"{save_path}_neutral.csv", 
        encoding="utf-8-sig",
        index=True
        )

if __name__ == "__main__":
    get_hrv_values(
        "./survey_preprocessed/lahv_1_normalized.csv", 
        MEANINGFUL_FEATURES, 
        "./survey_describe/lahv_1_normalized"
    )