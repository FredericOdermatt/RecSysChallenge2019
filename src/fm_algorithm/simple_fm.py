#fm
import numpy as np
from collections import Counter
import pandas as pd
from . import functions as f
from pathlib import Path
import click

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')
@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')

def main(data_path):
    print('start simple fm')
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
    subm_csv = data_directory.joinpath('submission_popular.csv')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    df_train = df_train.drop('timestamp',1)
    data = Counter(df_train.index).most_common(10000)
    keys = dict(data).keys()
    df_train = df_train[df_train.index.isin(keys)]

    df_train['_session_id'] = df_train.index

    print('simple fm finished')

if __name__ == '__main__':
    main()
