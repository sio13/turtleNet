import pandas as pd


def json_to_csv(json_path: str, csv_path: str):
    df_pandas = pd.read_json(open(json_path))

    df_pandas.to_csv(f"{csv_path}.csv")


if __name__ == '__main__':
    json_to_csv("results/json/data_augmentation/cifar_augmented.json", "results/csv/data_augmentation/cifar_augmented")
    json_to_csv("results/json/data_augmentation/cifar_not_augmented.json", "results/csv/data_augmentation/cifar_not_augmented")
    json_to_csv("results/json/data_augmentation/cifar_augmented_20_iter.json", "results/csv/data_augmentation/cifar_augmented_20_iter")

