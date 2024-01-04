import os

import pandas as pd

load_path = "./face"
save_path = "./face_filled_expression"
data_list = os.listdir(load_path)

for data in data_list:
    df = pd.read_csv(f"{load_path}/{data}")
    df.columns = ["frame", "basic", "valence"]

    df.loc[df["basic"] == 0, "valence"] = 0
    df.loc[df["basic"] == 1, "valence"] = 1
    df.loc[df["basic"] == 2, "valence"] = 2
    df.loc[df["basic"] == 3, "valence"] = 3
    df.loc[df["basic"] == 4, "valence"] = 4
    df.loc[df["basic"] == 5, "valence"] = 5
    df.loc[df["basic"] == 6, "valence"] = 6
    df.loc[df["basic"] == 7, "valence"] = 7

    df.loc[df["basic"] == " 0 ", "valence"] = 0
    df.loc[df["basic"] == " 1 ", "valence"] = 1
    df.loc[df["basic"] == " 2 ", "valence"] = 2
    df.loc[df["basic"] == " 3 ", "valence"] = 3
    df.loc[df["basic"] == " 4 ", "valence"] = 4
    df.loc[df["basic"] == " 5 ", "valence"] = 5
    df.loc[df["basic"] == " 6 ", "valence"] = 6
    df.loc[df["basic"] == " 7 ", "valence"] = 7
    df.loc[df["basic"] == " - ", "valence"] = 8
    
    df.loc[df["basic"] == "0 ", "valence"] = 0
    df.loc[df["basic"] == "1 ", "valence"] = 1
    df.loc[df["basic"] == "2 ", "valence"] = 2
    df.loc[df["basic"] == "3 ", "valence"] = 3
    df.loc[df["basic"] == "4 ", "valence"] = 4
    df.loc[df["basic"] == "5 ", "valence"] = 5
    df.loc[df["basic"] == "6 ", "valence"] = 6
    df.loc[df["basic"] == "7 ", "valence"] = 7
    df.loc[df["basic"] == "- ", "valence"] = 8
    
    df.loc[df["basic"] == " 0", "valence"] = 0
    df.loc[df["basic"] == " 1", "valence"] = 1
    df.loc[df["basic"] == " 2", "valence"] = 2
    df.loc[df["basic"] == " 3", "valence"] = 3
    df.loc[df["basic"] == " 4", "valence"] = 4
    df.loc[df["basic"] == " 5", "valence"] = 5
    df.loc[df["basic"] == " 6", "valence"] = 6
    df.loc[df["basic"] == " 7", "valence"] = 7
    df.loc[df["basic"] == " -", "valence"] = 8
    
    df.loc[df["valence"] == " - ", "valence"] = 8

    df =  df.astype({"valence" : "int"})
    df.to_csv(
        f"{save_path}/{data}", 
        encoding="utf-8-sig", 
        index=False,
    )
