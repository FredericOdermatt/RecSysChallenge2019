import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def reduce(df_items, splitting_point):

    #below better method to split data
    # df_train = df_items.iloc[:splitting_point]
    # df_test = df_items.iloc[splitting_point:]

    #TODO: REMOVE 100000 test data boundary
    df_items = np.split(df_items, [splitting_point,50000])
    df_train = df_items[0]
    df_test = df_items[1]

    train_reference = df_train['reference']
    test_reference = df_test['reference']
    df_train.drop(['reference'], axis=1, inplace=True)
    df_test.drop(['reference'], axis=1, inplace=True)

    print('Train data shape', df_train.shape)
    print('Test data shape', df_test.shape)
    
    #as we only have 1,0 values we actually dont need to scale I think
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
    encoding_dim = 20

    input_dim = Input(shape = (ncol, ))

    # Encoder Layers
    encoded1 = Dense(3000, activation = 'relu')(input_dim)
    encoded2 = Dense(2750, activation = 'relu')(encoded1)
    encoded3 = Dense(2500, activation = 'relu')(encoded2)
    encoded4 = Dense(2250, activation = 'relu')(encoded3)
    encoded5 = Dense(2000, activation = 'relu')(encoded4)
    encoded6 = Dense(1750, activation = 'relu')(encoded5)
    encoded7 = Dense(1500, activation = 'relu')(encoded6)
    encoded8 = Dense(1250, activation = 'relu')(encoded7)
    encoded9 = Dense(1000, activation = 'relu')(encoded8)
    encoded10 = Dense(750, activation = 'relu')(encoded9)
    encoded11 = Dense(500, activation = 'relu')(encoded10)
    encoded12 = Dense(250, activation = 'relu')(encoded11)
    encoded13 = Dense(encoding_dim, activation = 'relu')(encoded12)

    # Decoder Layers
    decoded1 = Dense(250, activation = 'relu')(encoded13)
    decoded2 = Dense(500, activation = 'relu')(decoded1)
    decoded3 = Dense(750, activation = 'relu')(decoded2)
    decoded4 = Dense(1000, activation = 'relu')(decoded3)
    decoded5 = Dense(1250, activation = 'relu')(decoded4)
    decoded6 = Dense(1500, activation = 'relu')(decoded5)
    decoded7 = Dense(1750, activation = 'relu')(decoded6)
    decoded8 = Dense(2000, activation = 'relu')(decoded7)
    decoded9 = Dense(2250, activation = 'relu')(decoded8)
    decoded10 = Dense(2500, activation = 'relu')(decoded9)
    decoded11 = Dense(2750, activation = 'relu')(decoded10)
    decoded12 = Dense(3000, activation = 'relu')(decoded11)
    decoded13 = Dense(ncol, activation = 'sigmoid')(decoded12)

    # Combine Encoder and Decoder layers
    autoencoder = Model(inputs = input_dim, outputs = decoded13)

    # Compile the Model
    autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

    autoencoder.summary()

    #TODO: different nb_epoch, prob region of 200-300 would be good
    autoencoder.fit(X_train, X_train, nb_epoch = 1, batch_size = 32, shuffle = False, 
                    validation_data = (X_test, X_test))    
    print("training completed")

    #Use Encoder level to reduce dimension of train and test data
    encoder = Model(inputs = input_dim, outputs = encoded13)
    encoded_input = Input(shape = (encoding_dim, ))
    print(f"encoding training data to {encoding_dim} dimensions...")
    #Predict the new train and test data using Encoder
    encoded_train = pd.DataFrame(encoder.predict(train_scaled))
    encoded_train = encoded_train.add_prefix('feature_')

    print(f"encoding test data to {encoding_dim} dimensions...")
    encoded_test = pd.DataFrame(encoder.predict(test_scaled))
    encoded_test = encoded_test.add_prefix('feature_')

    return encoded_train, encoded_test