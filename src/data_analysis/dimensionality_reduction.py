import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

def reduce(df_items, splitting_point):
    df_train = df_items.iloc[:splitting_point]
    df_test = df_items.iloc[splitting_point:]

    train_reference = df_train['reference']
    test_reference = df_test['reference']

    #print('train reference: ', train_reference)

    df_train.drop(['reference'], axis=1, inplace=True)
    df_test.drop(['reference'], axis=1, inplace=True)

    print('Train data shape', df_train.shape)
    print('Test data shape', df_test.shape)
    
    train_scaled = minmax_scale(df_train, axis = 0)
    test_scaled = minmax_scale(df_test, axis = 0)

    ncol = train_scaled.shape[1]

    #choose random 80% for training
    mask = np.random.choice(2,train_scaled.shape[0], p=[0.2,0.8])
    X_train = train_scaled[mask == 1]
    print('Xtrain data shape', X_train.shape)
    X_test = train_scaled[mask == 0]
    print('Xtest data shape', X_test.shape)

    #TODO: try different dimensions
    encoding_dim = 50
    
    return train_scaled