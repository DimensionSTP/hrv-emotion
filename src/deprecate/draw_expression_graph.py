import pandas as pd

df = pd.read_excel("facial_expression.xlsx")

survey_low_df = df[df["Valence"]<=3]
survey_neutral_df = df[df["Valence"]==4]
survey_high_df = df[df["Valence"]>=5]

low_count = survey_low_df.mode_idx4.value_counts()
neutral_count = survey_neutral_df.mode_idx4.value_counts()
high_count = survey_high_df.mode_idx4.value_counts()

low_count = low_count.sort_index()
neutral_count = neutral_count.sort_index()
high_count = high_count.sort_index()

print(len(survey_low_df))
print(len(survey_neutral_df))
print(len(survey_high_df))

print(low_count)
print(neutral_count)
print(high_count)