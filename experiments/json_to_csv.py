import pandas as pd
import sys
import os


def json_to_csv(json_path: str, csv_path: str):
    df_pandas = pd.read_json(open(json_path))

    df_pandas.to_csv(f"{csv_path}.csv")


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("no arguments")
        sys.exit(1)

    folder = sys.argv[1]
    files_to_parse = os.listdir(sys.argv[1])

    for filename in files_to_parse:
        partitioned_folder = folder.split('/')
        json_to_csv(f"{folder}/{filename}", f"{partitioned_folder[0]}/csv/{partitioned_folder[2]}/{filename}")

