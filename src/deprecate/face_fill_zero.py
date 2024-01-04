import os

import pandas as pd

load_path = "./face"
save_path = "./face_filled"
data_list = os.listdir(load_path)

for data in data_list:
    df = pd.read_csv(f"{load_path}/{data}")
    df.columns = ["frame", "basic", "valence"]

    df.loc[df["basic"] == 0, "valence"] = 1
    df.loc[df["basic"] == 1, "valence"] = 2
    df.loc[df["basic"] == 2, "valence"] = 0
    df.loc[df["basic"] == 3, "valence"] = 1
    df.loc[df["basic"] == 4, "valence"] = 0
    df.loc[df["basic"] == 5, "valence"] = 0
    df.loc[df["basic"] == 6, "valence"] = 0
    df.loc[df["basic"] == 7, "valence"] = 1

    df.loc[df["basic"] == " 0 ", "valence"] = 1
    df.loc[df["basic"] == " 1 ", "valence"] = 2
    df.loc[df["basic"] == " 2 ", "valence"] = 0
    df.loc[df["basic"] == " 3 ", "valence"] = 1
    df.loc[df["basic"] == " 4 ", "valence"] = 0
    df.loc[df["basic"] == " 5 ", "valence"] = 0
    df.loc[df["basic"] == " 6 ", "valence"] = 0
    df.loc[df["basic"] == " 7 ", "valence"] = 1
    df.loc[df["basic"] == " - ", "valence"] = 3
    
    df.loc[df["basic"] == "0 ", "valence"] = 1
    df.loc[df["basic"] == "1 ", "valence"] = 2
    df.loc[df["basic"] == "2 ", "valence"] = 0
    df.loc[df["basic"] == "3 ", "valence"] = 1
    df.loc[df["basic"] == "4 ", "valence"] = 0
    df.loc[df["basic"] == "5 ", "valence"] = 0
    df.loc[df["basic"] == "6 ", "valence"] = 0
    df.loc[df["basic"] == "7 ", "valence"] = 1
    df.loc[df["basic"] == "- ", "valence"] = 3
    
    df.loc[df["basic"] == " 0", "valence"] = 1
    df.loc[df["basic"] == " 1", "valence"] = 2
    df.loc[df["basic"] == " 2", "valence"] = 0
    df.loc[df["basic"] == " 3", "valence"] = 1
    df.loc[df["basic"] == " 4", "valence"] = 0
    df.loc[df["basic"] == " 5", "valence"] = 0
    df.loc[df["basic"] == " 6", "valence"] = 0
    df.loc[df["basic"] == " 7", "valence"] = 1
    df.loc[df["basic"] == " -", "valence"] = 3
    
    df.loc[df["valence"] == " - ", "valence"] = 3

    df =  df.astype({"valence" : "int"})
    df.to_csv(
        f"{save_path}/{data}", 
        encoding="utf-8-sig", 
        index=False,
    )
