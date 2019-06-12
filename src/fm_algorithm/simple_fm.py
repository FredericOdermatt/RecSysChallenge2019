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
    subm_csv = data_directory.joinpath('submission_ffm.csv')

    meta_encoded_csv = data_directory.joinpath('item_metadata_encoded.csv')
    training_ffm = data_directory.joinpath('ffm_out/training_ffm.txt')
    test_ffm = data_directory.joinpath('ffm_out/test_ffm.txt')
    val_ffm = data_directory.joinpath('ffm_out/val_ffm.txt')
    model_ffm = data_directory.joinpath('ffm_out/model.out')
    output_ffm = data_directory.joinpath('ffm_out/output.txt')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv,nrows=100000)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    print("Preprocessing training dataset ...")
    mask = df_train['action_type'] == 'clickout item'
    df_clicks = df_train[mask]
    df_clicks = df_clicks.head(1000)

    df_clicks = df_clicks[['session_id','reference','impressions','platform','city']]

    implist = df_clicks['impressions'].values.tolist()
    implist = f.getlist(implist)
    df_clicks = df_clicks.drop('impressions',1)
    df_clicks['impressions'] = implist
    df_clicks['impressions'] = df_clicks.apply(lambda x: list(set(x['impressions'])-set([x['reference']])), axis=1)
    df_unclicked = f.explode(df_clicks,'impressions')
    df_unclicked = df_unclicked.drop('reference',1)
    df_unclicked = df_unclicked.rename(columns={'impressions': 'reference'})
    df_unclicked = df_unclicked.drop_duplicates()
    df_clicks['reference'] = df_clicks.apply(lambda x: list([x['reference']]), axis=1)
    df_clicks = df_clicks.drop('impressions',1)
    df_clicks = f.explode(df_clicks,'reference')
    df_clicks = df_clicks.drop_duplicates()
    df_clicks = resample(df_clicks,replace=True,n_samples=(int)(df_unclicked.shape[0]/2.))
    df_clicks['label'] = 1
    df_unclicked['label'] = 0

    print("Preprocessing test dataset ...")
    df_target = f.get_submission_target(df_test)
    print('Number of test dataset : %d' % df_target.shape[0])
    # df_target = df_target.head(10000)
    df_target = df_target[['user_id','session_id','timestamp','step','reference','impressions','platform','city']]
    implist = df_target['impressions'].values.tolist()
    implist = f.getlist(implist)
    df_target = df_target.drop('impressions',1)
    df_target['impressions'] = implist
    df_target = f.explode(df_target,'impressions')
    df_target = df_target.drop('reference',1)
    df_target = df_target.rename(columns={'impressions': 'reference'})

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

    df_merged = pd.concat([df_clicks,df_unclicked],ignore_index=True)
    df_merged = df_merged.sort_values(by=['reference'])
    df_merged = df_merged.merge(df_meta,how='left',on='reference')
    df_merged, listkeys = f.onehotsession(df_merged,['platform','city','session_id'])
    listkeys.insert(0,propkeys)
    df_merged = df_merged.drop('reference',1)
    df_merged = df_merged.fillna(0)
    print(df_merged)
    # print(df_merged[df_merged.isnull().any(axis=1)])

    print("Encoding test dataset ...")
    df_target = df_target.sort_values(by=['reference'])
    df_target = df_target.merge(df_meta,how='left',on='reference')
    df_target = df_target.fillna(0)
    df_tarref = df_target[['user_id','reference','session_id','timestamp','step']]
    df_target = df_target.drop(['user_id','reference','timestamp','step'],1)
    print('Number of encoded test dataset : %d' % df_target.shape[0])
    df_target = f.fittarget(df_target,listkeys,['platform','city','session_id'])
    df_target['label'] = 0

    y = df_merged['label']
    # df_merged = df_merged.drop('label',1)
    fieldmap = f.getfieldmap(listkeys)
    fieldmap = pd.DataFrame(fieldmap, index=[0])

    x_train, x_val, y_train, y_val = train_test_split(df_merged,y,test_size = 0.2)

    print("Writing FFM input ...")
    # features.remove('session_id')
    f.writeffm(x_train,listkeys,training_ffm.as_posix())
    f.writeffm(df_target,listkeys,test_ffm.as_posix())
    f.writeffm(x_val,listkeys,val_ffm.as_posix())

    import xlearn as xl

    # x_train = xl.DMatrix(x_train,y_train,fieldmap)
    # x_test = xl.DMatrix(x_test,y_test,fieldmap)

    try:
        model_ffm.resolve(strict=True)
    except FileNotFoundError:
        print("Training FFM ...")
        ffm_model = xl.create_ffm()
        ffm_model.setTrain(training_ffm.as_posix())
        ffm_model.setValidate(val_ffm.as_posix())
        # ffm_model.setTrain(x_train)
        param = {'task':'binary','lr':0.2,'lambda':0.001,'metric':'f1','opt':'adagrad','k':4,'epoch':10,'init':0.33}

        ffm_model.fit(param,model_ffm.as_posix())
        ffm_model.cv(param)
    else:
        print("Model file found")
        ffm_model = xl.create_ffm()

    ffm_model.setSigmoid()
    ffm_model.setTest(test_ffm.as_posix())

    # ffm_model.setTest(x_test)

    ffm_model.predict(model_ffm.as_posix(),output_ffm.as_posix())

    print(f"Writing {subm_csv}...")

    df_predict = pd.read_csv(output_ffm.as_posix(),names=['predict'])
    df_tarref = df_tarref.reset_index(drop=True)
    df_predict = pd.concat([df_tarref,df_predict],axis=1)
    df_predict = f.group_concat(df_predict,['user_id','session_id','timestamp','step'],'reference')
    df_predict = df_predict.rename(columns={'reference': 'item_recommendations'})
    print(df_predict)
    df_predict.to_csv(subm_csv, index=False)

    print('Finished')

if __name__ == '__main__':
    main()
