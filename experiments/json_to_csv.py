import pandas as pd


def json_to_csv(json_path: str, csv_path: str):
    df_pandas = pd.read_json(open(json_path))

    df_pandas.to_csv(f"{csv_path}.csv")


if __name__ == '__main__':
    json_to_csv("results/json/compare_damage/cifar_natural_trained", "results/csv/compare_damage/cifar_natural_trained")
    json_to_csv("results/json/compare_damage/mnist_natural_trained", "results/csv/compare_damage/mnist_natural_trained")