import pandas as pd


def json_to_csv(json_path: str, csv_path: str):
    df_pandas = pd.read_json(open(json_path))

    df_pandas.to_csv(f"{csv_path}.csv")


if __name__ == '__main__':
    json_to_csv(
        "results/json/data_augmentation/cifar_natural_trained_0.1.json",
        "results/csv/data_augmentation/cifar_natural_trained_0_1")
    json_to_csv(
        "results/json/data_augmentation/cifar_natural_trained_0.2.json",
        "results/csv/data_augmentation/cifar_natural_trained_0_2")
    json_to_csv(
        "results/json/data_augmentation/cifar_natural_trained_0.3.json",
        "results/csv/data_augmentation/cifar_natural_trained_0_3")

    json_to_csv(
        "results/json/data_augmentation/mnist_natural_trained_0.1.json",
        "results/csv/data_augmentation/mnist_natural_trained_0_1")
    json_to_csv(
        "results/json/data_augmentation/mnist_natural_trained_0.2.json",
        "results/csv/data_augmentation/mnist_natural_trained_0_2")
    json_to_csv(
        "results/json/data_augmentation/mnist_natural_trained_0.3.json",
        "results/csv/data_augmentation/mnist_natural_trained_0_3.json")
