from pandas import read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


from keras.models import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

path='../data/iBeacon_RSSI_Unlabeled.csv'
x_un = read_csv(path, index_col=None)
x_un.drop(["location", "date"], axis = 1, inplace = True)
x_un = (x_un+200)/200

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
input_layer = Input(shape=(x_un.shape[1],))
enc = Dense(10, activation='relu')(input_layer)
enc = Dense(5, activation='relu')(enc)
dec = Dense(5, activation='relu')(enc)
dec = Dense(10, activation='relu')(dec)
output_layer = Dense(x_un.shape[1], activation='relu')(dec)

model = Model(input_layer, output_layer)
model.compile(optimizer=Adam(.001),
              loss=rmse,
              metrics=['mse'])
hist = model.fit(x_un, x_un, epochs=20, batch_size=10, verbose=2)

path='../data/iBeacon_RSSI_Labeled.csv'
a = read_csv(path, index_col=None)
#a.drop(["date"], axis = 1, inplace = True)
x_aug = a.groupby('location').filter(lambda b : (len(b) < 10) & (len(b) > 1))
x_aug = x_aug.reset_index(drop=True)
x_aug_1 = (x_aug.drop(['location','date'], axis = 1) + 200)/200
preds = model.predict(x_aug_1)
pred = (-200 + preds*200).astype(int)
pred[pred<-130] = -200
df = pd.DataFrame(pred)
df.columns = x_aug_1.columns
x_aug.iloc[:,2:] = df.iloc[:,:]
x_aug = x_aug[~(x_aug.iloc[:,2:] == -200).all(axis=1)]
x_aug = x_aug.reset_index(drop=True)
x_aug.to_csv('../data/iBeacon_RSSI_Labeled_aug_stdenc.csv',index = False)
