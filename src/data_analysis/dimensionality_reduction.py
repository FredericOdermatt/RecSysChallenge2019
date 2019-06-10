import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

def reduce(df_items, splitting_point, encoding_dim, nmbs_epoch):

    # df_items = df_items.drop('reference',axis=1)
    df_train = np.split(df_items, [splitting_point])[0]

    ncol = df_train.shape[1]

    #choose random 80% for training
    mask = np.random.choice(2,df_train.shape[0], p=[0.2,0.8])
    X_train = df_train[mask == 1]
    X_test = df_train[mask == 0]

    input_dim = Input(shape = (ncol, ))

    # Encoder Layers
    encoded1 = Dense(100, activation = 'relu')(input_dim)
    encoded2 = Dense(50, activation = 'relu')(encoded1)
    encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)

    # Decoder Layers
    decoded1 = Dense(50, activation = 'relu')(encoded3)
    decoded2 = Dense(100, activation = 'relu')(decoded1)
    decoded3 = Dense(ncol, activation = 'sigmoid')(decoded2)

    # Combine Encoder and Decoder layers
    autoencoder = Model(inputs = input_dim, outputs = decoded3)

    # Compile the Model
    opt = optimizers.Adagrad(lr=0.1)
    autoencoder.compile(optimizer = opt, loss = 'binary_crossentropy')

    autoencoder.summary()

    autoencoder.fit(X_train, X_train, nb_epoch = nmbs_epoch, batch_size = 32,
                    shuffle = False, validation_data = (X_test, X_test))
    print("Training completed")

    #Use Encoder level to reduce dimension of train and test data
    encoder = Model(inputs = input_dim, outputs = encoded3)
    encoded_input = Input(shape = (encoding_dim, ))

    print(f"Encoding item features to {encoding_dim} dimensions... (this takes a while)")
    #Predict the new train and test data using Encoder
    encoded_item = pd.DataFrame(encoder.predict(df_items))
    encoded_item = encoded_item.add_prefix('feature_')

    return encoded_item


def undoonehot(row):
    rating_keys = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars']
    rate = [0,1,2,3,4,1,2,3]
    for i, key in enumerate(rating_keys):
        if row.loc[key]==1:
            return rate[i]

    return row
