import pandas as pd

MEANINGFUL_FEATURES = ["rMSSD", "HF", "lnLF", "lnHF", "RSA_PB"]

def get_hrv_values(load_path, meaningful_features, save_path):
    hrv_data = pd.read_csv(load_path)
    hrv_data = hrv_data[meaningful_features]
    hrv_values = hrv_data.describe()
    hrv_values.to_csv(
        f"{save_path}", 
        encoding="utf-8-sig",
        index=True
        )

if __name__ == "__main__":
    get_hrv_values(
        "./preprocessed/low_arousal_1_normalized.csv", 
        MEANINGFUL_FEATURES, 
        "./describe/low_arousal_1_normalized_describe.csv"
    )