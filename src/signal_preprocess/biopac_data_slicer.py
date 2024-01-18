import os
from datetime import datetime
from typing import List

from tqdm import tqdm
import pandas as pd


class BioPacDataSlicer():
    def __init__(
        self, 
        log_path: str, 
        data_path: str, 
        save_path: str,
    ) -> None:
        self.log_path = log_path
        self.data_path = data_path
        self.save_path = save_path
        self.log_file_names = sorted(os.listdir(log_path))
        self.data_file_names = sorted(os.listdir(data_path))
    
    def __call__(self) -> None:
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        for i in tqdm(range(50)):
            log_file_name = self.log_file_names[i]
            data_file_name = self.data_file_names[i]
            video_logs = self.get_video_logs(log_file_name)
            ecg_start_time = self.get_ecg_start_time(data_file_name)
            absolute_time_stamp = self.get_absolute_time_stamp(video_logs, ecg_start_time)
            stimuli_types = self.get_stimuli_types(video_logs)
            data_file = self.get_data_file(data_file_name)
            subject_name = self.get_subject_name(log_file_name, data_file_name)
            for i in range(5):
                data_boundary = data_file[(data_file[0]>=float(absolute_time_stamp[i])) & (data_file[0]<float(absolute_time_stamp[i])+240)]
                data_boundary = data_boundary.iloc[:120000]
                stimulus_type = stimuli_types[i]
                data_boundary.to_csv(
                    f"{self.save_path}/{subject_name}_{stimulus_type}.csv", 
                    encoding="utf-8-sig", 
                    index=False,
                    )
    
    def get_ecg_start_time(self, data_file_name: str,) -> str:
        file = open(f"{self.data_path}/{data_file_name}", "r")
        i = 0
        while True:
            line = file.readline()
            i = i + 1
            if i == 11:
                ecg_start_time = line.split("\t")[1][:-14]
                break
        return ecg_start_time
    
    def get_video_logs(self, log_file_name: str,) -> List[str]:
        file = open(f"{self.log_path}/{log_file_name}", "r")
        video_logs = []
        while True:
            line = file.readline()
            if not line:
                break
            if "Reference" in line:
                video_logs.append(line)
            if "./videos" in line:
                video_logs.append(line)
        return video_logs
    
    def get_absolute_time_stamp(
        self, 
        video_logs: List[str], 
        ecg_start_time: str,
    ) -> List[float]:
        datetimes = []
        datetime_format = "%H:%M:%S.%f"
        
        for i, video_log in enumerate(video_logs):
            t = video_log.split(" ")[1][:-1]
            datetime_t = datetime.strptime(t, datetime_format)
            datetimes.append(datetime_t)
        
        ecg_start_time = datetime.strptime(ecg_start_time, datetime_format)
        
        absolute_time_stamp = []
        
        for i in range(5):
            time_stamp = datetimes[i] - ecg_start_time
            time_stamp_second = "%0.3f" % round(time_stamp.total_seconds(), 2)
            absolute_time_stamp.append(time_stamp_second)
        return absolute_time_stamp
    
    def get_stimuli_types(self, video_logs: List[str],) -> List[str]:
        stimuli = ["reference"]
        for i in range(1, 5):
            stimuli.append(video_logs[i][9:13])
        return stimuli
    
    def get_data_file(self, data_file_name: str,) -> pd.DataFrame:
        data_file = pd.read_csv(
            f"{self.data_path}/{data_file_name}", header=None, delimiter="\t", skiprows=23
        )
        return data_file
    
    def get_subject_name(
        self, 
        log_file_name: str, 
        data_file_name: str,
    ) -> str:
        log_file_subject_name = log_file_name[:3]
        data_file_subject_name = data_file_name[:3]
        if log_file_subject_name == data_file_subject_name:
            subject_name = log_file_subject_name
        else:
            raise Exception("log file subject name doesn't match with data file subject name.")
        return subject_name