#fm
import numpy as np
import pandas as pd
from . import functions as f
from pathlib import Path
import click
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')
@click.command()
@click.option('--data-path', default='./data/', help='Directory for the CSV files')

def main(data_path):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
    meta_csv = data_directory.joinpath('item_metadata.csv')
    subm_csv = data_directory.joinpath('submission_popular.csv')

    meta_encoded_csv = data_directory.joinpath('item_metadata_encoded.csv')
    training_ffm = data_directory.joinpath('training_ffm.txt')
    test_ffm = data_directory.joinpath('test_ffm.txt')
    model_ffm = data_directory.joinpath('model.out')
    output_ffm = data_directory.joinpath('output.txt')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv,nrows=10000)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv,nrows=10000)

    print("Preprocessing sessions ...")
    mask = df_train['action_type'] == 'clickout item'
    df_clicks = df_train[mask]

    df_clicks = df_clicks[['session_id','reference','impressions','platform','city']]

    implist = df_clicks['impressions'].values.tolist()
    implist = f.getlist(implist)
    implist = pd.Series(implist)
    df_clicks = df_clicks.drop('impressions',1)
    df_clicks['impressions'] = implist.values
    # df_clicks = df_clicks.groupby('session_id',as_index=False).agg(lambda x: list(x))
    # df_clicks['impressions'] = df_clicks['impressions'].apply(lambda x: sum(x,[]))
    df_clicks['impressions'] = df_clicks.apply(lambda x: list(set(x['impressions'])-set([x['reference']])), axis=1)
    df_unclicked = f.flatten(df_clicks,['session_id','platform','city'],'impressions')
    df_unclicked = df_unclicked.astype({'reference':int})
    df_clicks['reference'] = df_clicks.apply(lambda x: list([x['reference']]), axis=1)
    df_clicks = f.flatten(df_clicks,['session_id','platform','city'],'reference')
    df_clicks = df_clicks.astype({'reference':int})
    df_clicks = resample(df_clicks,replace=True,n_samples=df_unclicked.shape[0])
    df_clicks['label'] = 1
    df_unclicked['label'] = 0

    # df_clicks.reset_index(inplace=True,drop=True)
    # df_unclicked.reset_index(inplace=True,drop=True)

    try:
        meta_encoded_csv.resolve(strict=True)
    except FileNotFoundError:
        print(f"Reading {meta_csv} ...")
        df_meta = pd.read_csv(meta_csv)
        print("Preprocessing metadata ...")
        df_metaonehot, propkeys = f.onehotprop(df_meta['properties'])
        df_meta = df_meta['item_id'].to_frame().join(df_metaonehot)
        df_meta = df_meta.rename(columns={'item_id':'reference'})
        df_meta.to_csv(meta_encoded_csv,index=None,header=True)
    else:
        print("Encoded metadata file found")
        print(f"Reading {meta_encoded_csv} ...")
        df_meta = pd.read_csv(meta_encoded_csv)
        propkeys = df_meta.columns.values.tolist()
        propkeys.remove('reference')
        propkeys = dict(zip(propkeys,range(len(propkeys))))

    print("Merging dataframes ...")

    df_merged = pd.concat([df_clicks,df_unclicked])
    df_merged = df_merged.sort_values(by=['reference'])
    df_merged = df_merged.merge(df_meta,how='left',on='reference')
    df_merged = df_merged.dropna()
    df_merged, listkeys = f.onehotsession(df_merged,['platform','city'])
    listkeys.insert(0,propkeys)
    df_merged = df_merged.drop('reference',1)

    print(df_merged)

    x_train, x_test = train_test_split(df_merged,test_size = 0.3)

    print("Writing FFM input ...")
    # features.remove('session_id')
    f.writeffm(x_train,listkeys,training_ffm.as_posix())
    f.writeffm(x_test,listkeys,test_ffm.as_posix())

    import xlearn as xl

    print("Training FFM ...")
    ffm_model = xl.create_ffm()
    ffm_model.setTrain(training_ffm.as_posix())
    param = {'task':'binary','lr':0.2,'lambda':0.001,'metric':'acc','opt':'adagrad','k':5,'epoch':20}

    ffm_model.fit(param,model_ffm.as_posix())
    ffm_model.cv(param)

    ffm_model.setTest(test_ffm.as_posix())

    ffm_model.setSigmoid()
    ffm_model.predict(model_ffm.as_posix(),output_ffm.as_posix())

    print('Finished')

if __name__ == '__main__':
    main()
