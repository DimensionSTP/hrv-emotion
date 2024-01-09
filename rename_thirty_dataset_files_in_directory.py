import os

from tqdm import tqdm

def rename_thirty_dataset_files_in_directory(directory, mapping):
    # 디렉토리 내의 모든 파일을 순회
    for filename in tqdm(os.listdir(directory)):
        parts = filename.split('_')
        if len(parts) == 3:
            name, tag = parts[1], parts[2].split('.')[0]

            # 매핑에 따라 새 태그 설정
            if tag in mapping:
                new_tag = mapping[tag]
                new_filename = f"{name}_{new_tag}.txt"
                # 파일 이름 변경
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


if __name__ == "__main__":
    # 사용 예시
    DIRECTORY = "./dataset/thirty/signals"
    MAPPING = {"1": "HALV", "2": "HAHV", "3": "LAHV", "4": "LALV", "Reference": "reference"}
    rename_thirty_dataset_files_in_directory(DIRECTORY, MAPPING)