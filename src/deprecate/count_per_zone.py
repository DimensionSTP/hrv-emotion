import numpy as np
import pandas as pd

def count_per_zone(lnlf_minimal_boundary, lnlf_maximal_boundary, lnhf_minimal_boundary, lnhf_maximal_boundary):
    df = pd.read_excel("survey.xlsx")

    df["stimuli_arousal"] = np.nan
    df.loc[df["Arousal"] >= 5, "stimuli_arousal"] = "각성"
    df.loc[df["Arousal"] == 4, "stimuli_arousal"] = "중립"
    df.loc[df["Arousal"] <= 3, "stimuli_arousal"] = "이완"
    
    zone1 = df[(df["lnLF"]<lnlf_minimal_boundary) & (df["lnHF"]>=lnhf_maximal_boundary)]
    zone2 = df[(df["lnLF"]>=lnlf_minimal_boundary) & (df["lnLF"]<lnlf_maximal_boundary) & (df["lnHF"]>=lnhf_maximal_boundary)]
    zone3 = df[(df["lnLF"]>=lnlf_maximal_boundary) & (df["lnHF"]>=lnhf_maximal_boundary)]
    zone8 = df[(df["lnLF"]<lnlf_minimal_boundary) & (df["lnHF"]>=lnhf_minimal_boundary) & (df["lnHF"]<lnhf_maximal_boundary)]
    zone_ref = df[(df["lnLF"]>=lnlf_minimal_boundary) & (df["lnLF"]<lnlf_maximal_boundary) & (df["lnHF"]>=lnhf_minimal_boundary) & (df["lnHF"]<lnhf_maximal_boundary)]
    zone4 = df[(df["lnLF"]>=lnlf_maximal_boundary) & (df["lnHF"]>=lnhf_minimal_boundary) & (df["lnHF"]<lnhf_maximal_boundary)]
    zone7 = df[(df["lnLF"]<lnlf_minimal_boundary) & (df["lnHF"]<lnhf_minimal_boundary)]
    zone6 = df[(df["lnLF"]>=lnlf_minimal_boundary) & (df["lnLF"]<lnlf_maximal_boundary) & (df["lnHF"]<lnhf_minimal_boundary)]
    zone5 = df[(df["lnLF"]>=lnlf_maximal_boundary) & (df["lnHF"]<lnhf_minimal_boundary)]
    
    zones = [zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, zone_ref]
    zones_count = []
    
    for zone in zones:
        zone = pd.DataFrame(zone["stimuli_arousal"].value_counts())
        zone.columns = ["분포 수"]
        print(zone)
        if zone.empty:
            zone = pd.DataFrame(
                {
                    "분포 수" : [0, 0, 0], 
                    "백분율(%)" : [0, 0, 0]
                }, 
                index = ["각성", "중립", "이완"]
            )
            print(zone)
        else:
            if "각성" not in zone.T:
                zone.loc["각성"] = [0]
            if "중립" not in zone.T:
                zone.loc["중립"] = [0]
            if "이완" not in zone.T:
                zone.loc["이완"] = [0]
            zone["백분율(%)"] = np.nan
            zone["백분율(%)"]["각성"] = round(zone["분포 수"]["각성"]/94*100, 2)
            zone["백분율(%)"]["중립"] = round(zone["분포 수"]["중립"]/34*100, 2)
            zone["백분율(%)"]["이완"] = round(zone["분포 수"]["이완"]/72*100, 2)
            print(zone)
        zones_count.append(zone)
    
    for i, zone_count in enumerate(zones_count):
        if i+1 == 9:
            zone_count.to_csv(
                f"./tmp/{lnlf_minimal_boundary}_{lnlf_maximal_boundary}_{lnhf_minimal_boundary}_{lnhf_maximal_boundary}/zone_reference.csv",
                encoding="utf-8-sig", 
                index=True,
            )
        else:
            zone_count.to_csv(
                f"./tmp/{lnlf_minimal_boundary}_{lnlf_maximal_boundary}_{lnhf_minimal_boundary}_{lnhf_maximal_boundary}/zone{i+1}.csv",
                encoding="utf-8-sig", 
                index=True,
            )

if __name__ == "__main__":
    count_per_zone(5, 6.5, 4.3, 6.8)                    