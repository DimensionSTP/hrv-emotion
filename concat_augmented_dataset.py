import pandas as pd


def concat_augmented_dataset(dataset_path: str, dataset_name1: str, dataset_name2: str, save_name: str,) -> None:
    df1 = pd.read_excel(f"{dataset_path}/{dataset_name1}")
    df2 = pd.read_excel(f"{dataset_path}/{dataset_name2}")
    concat_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    concat_df.to_excel(
        f"{dataset_path}/{save_name}",
        sheet_name="normalized",
        index=False,
    )


if __name__ == "__main__":
    concat_augmented_dataset(
        dataset_path="./tabular_dataset", 
        dataset_name1="sl_augmented.xlsx",
        dataset_name2="thirty_augmented.xlsx",
        save_name="sl_thirty_augmented.xlsx"
    )