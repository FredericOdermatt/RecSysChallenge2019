#fm
import numpy as np
import pandas as pd
from . import functions as f
from pathlib import Path
import click

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')
@click.command()
@click.option('--data-path', default='./data/', help='Directory for the CSV files')

def main(data_path):
    print('start simple fm')
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
    meta_csv = data_directory.joinpath('item_metadata.csv')
    subm_csv = data_directory.joinpath('submission_popular.csv')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv,nrows=10000)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv,nrows=10000)
    print(f"Reading {meta_csv} ...")
    df_meta = pd.read_csv(meta_csv,nrows=100000)
    print("Reading finished")

    mask = df_train['action_type'] == 'clickout item'
    df_clicks = df_train[mask]

    df_clicks = df_clicks[['session_id','reference','impressions']]
    df_clicks = df_clicks.astype({'reference':int})

    implist = df_clicks['impressions'].values.tolist()
    implist = f.getlist(implist)
    implist = pd.Series(implist)
    df_clicks = df_clicks.drop('impressions',1)
    df_clicks['impressions'] = implist.values
    df_clicks = df_clicks.groupby('session_id',as_index=False).agg(lambda x: list(x))
    df_clicks['impressions'] = df_clicks['impressions'].apply(lambda x: sum(x,[]))
    df_clicks['impressions'] = df_clicks.apply(lambda x: list(set(x['impressions'])-set(x['reference'])), axis=1)
    df_unclicked = f.flatten(df_clicks,'session_id','impressions')
    df_clicks = f.flatten(df_clicks,'session_id','reference')
    df_clicks['label'] = 1
    df_unclicked['label'] = 0

    df_metaonehot = f.onehotencode(df_meta['properties'])
    df_meta = df_meta['item_id'].to_frame().join(df_metaonehot)
    df_meta = df_meta.rename(columns={'item_id':'reference'})

    df_merged = pd.concat([df_clicks,df_unclicked])
    df_merged = pd.merge(df_merged, df_meta, on='reference')

    print(df_merged)

    # next step: transform city to int by counting # of cities without duplication
    # add metadata of references
    # put it into ffm

    # df_impprice = df_train[['impressions','prices']]
    # df_impprice = df_impprice.dropna()
    # print(df_impprice.head())
    # #
    # pricelist = df_impprice['prices'].values.tolist()
    # for i,textin in enumerate(pricelist): pricelist[i] = f.regextostr(textin)
    # pricelist = sum(pricelist,[])
    # pricelist = np.asarray(pricelist,dtype=np.int32)
    # print(pricelist)
    # print(r'Maximum price is : %d' % pricelist.max())
    #
    # n, bins, patches = plt.hist(pricelist,100)
    # plt.xlabel('price')
    # plt.ylabel('number')
    # plt.title('prices')
    # plt.grid(True)
    # plt.draw()
    # plt.savefig('./prices.png', dpi=300, bbox_inches='tight')

    # print(pricelist)

    # df_train = df_train.drop('timestamp',1)

    # data = Counter(df_train.index).most_common(10000)
    # keys = dict(data).keys()
    # df_train = df_train[df_train.index.isin(keys)]
    #
    # # print(df_train.head())
    #
    # df_train['_session_id'] = df_train.index
    #
    # tf_train = pd.get_dummies(df_train)
    #
    # # print("transform finished")
    # # print(tf_train.head())
    #
    # filtered_train = tf_train.filter(regex="impressions.*|")
    #
    # accumulated = filtered_train.groupby(filtered_train.index).sum()
    # accumulated = filtered_train.rename(columns=lambda column_name: 'accumulated_impression' +  column_name)
    #
    # merged = pd.merge(tf_train,accumulated,left_index=True,right_index=True)





















    print('simple fm finished')

if __name__ == '__main__':
    main()
