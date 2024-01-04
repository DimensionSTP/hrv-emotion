import os
import pandas as pd

# 폴더가 있는 경로를 지정합니다.
folder_path = "./data_all/"

# 해당 경로의 파일 리스트를 불러옵니다.
file_list = os.listdir(folder_path)

# ".txt" 확장자를 포함하지 않은 파일 이름들을 저장할 리스트 생성
name_list = []

# 파일 리스트에서 ".txt" 확장자를 제외하고 이름만 추출하여 리스트에 추가
for file_name in file_list:
    if file_name.endswith(".txt"):
        name_list.append(file_name[:-4])

# 이름을 알파벳 순으로 정렬합니다.
name_list.sort()

# 결과를 출력합니다.
# print(name_list)

# 사람 이름과 인덱스를 매핑할 딕셔너리를 생성합니다.
name_to_index_mapping = {}

# 리스트를 순회하면서 이름과 인덱스를 딕셔너리에 매핑합니다.
for index, name in enumerate(name_list):
    name_to_index_mapping[index] = name

# 결과를 출력합니다.
# print(name_to_index_mapping)

df = pd.read_csv("./stimulus_preprocessed/reference_1_raw.csv")
df["new_tPow"] = df["tPow"] - df["VLF"]

# 기준_칼럼을 기준으로 데이터프레임을 정렬합니다.
df_sorted = df.sort_values(by="new_tPow")

# 상위 25개와 하위 25개의 인덱스를 구합니다.
top_25_indexes = df_sorted.head(25).index
bottom_25_indexes = df_sorted.tail(25).index

# 결과를 출력합니다.
# print("상위 25개의 인덱스:", top_25_indexes)
# print("하위 25개의 인덱스:", bottom_25_indexes)

# 특정 숫자들에 해당하는 사람 이름을 저장할 리스트 생성
top25_people = []

# 원하는 숫자들에 해당하는 사람 이름을 리스트에 추가
for number in top_25_indexes:
    if number in name_to_index_mapping:
        top25_people.append(name_to_index_mapping[number])
        
# 특정 숫자들에 해당하는 사람 이름을 저장할 리스트 생성
tail25_people = []

# 원하는 숫자들에 해당하는 사람 이름을 리스트에 추가
for number in bottom_25_indexes:
    if number in name_to_index_mapping:
        tail25_people.append(name_to_index_mapping[number])
        
print(top25_people)
print(tail25_people)