import os

import pandas as pd
import numpy as np


def calc_rmse_mae(
    rhrv_path: str, hrv_path: str, emotion: str, save_path: str
):
    rhrv_files = os.listdir(f"{rhrv_path}/{emotion}")
    hrv_files = os.listdir(f"{hrv_path}/{emotion}")
    rmses = []
    maes = []
    if len(rhrv_files) == len(hrv_files):
        for i in range(len(rhrv_files)):
            if rhrv_files[i][:3] == hrv_files[i][:3]:
                rhrv_data = pd.read_csv(f"{rhrv_path}/{emotion}/{rhrv_files[i]}")
                hrv_data = pd.read_csv(f"{hrv_path}/{emotion}/{hrv_files[i]}")
                rmse = np.sqrt(((rhrv_data - hrv_data) ** 2).mean())
                mae = np.abs(rhrv_data - hrv_data).mean()
                rmses.append(rmse)
                maes.append(mae)
            else:
                raise Exception("rHRV file subject name doesn't match with HRV file subject name.")
    else:
        raise Exception("number of rHRV files doesn't match with HRV files.")
    pd.DataFrame(rmses).to_csv(f"{save_path}/rmse/rmse_{emotion}.csv", index=False)
    pd.DataFrame(maes).to_csv(f"{save_path}/mae/mae_{emotion}.csv", index=False)


if __name__ == "__main__":
    COMMON_PATH = "emotions/raw"
    RHRV_PATH = f"./dataset/sl_rppg/{COMMON_PATH}"
    HRV_PATH = f"./dataset/sl/{COMMON_PATH}"
    EMOTIONS = ["hahv", "halv", "lahv", "lalv", "reference"]
    SAVE_PATH = "./dataset/differences"
    for emotion in EMOTIONS:
        calc_rmse_mae(
            rhrv_path=RHRV_PATH,
            hrv_path=HRV_PATH,
            emotion=emotion,
            save_path=SAVE_PATH
        )