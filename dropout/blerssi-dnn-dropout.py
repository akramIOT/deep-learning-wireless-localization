%matplotlib inline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def fix_pos(x_cord):
    x = 87 - ord(x_cord.upper())
    return x

def l2_dist(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dx = dx ** 2
    dy = dy ** 2
    dists = dx + dy
    dists = np.sqrt(dists)
    return np.mean(dists), dists

def create_deep(inp_dim,dropout):
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dropout(dropout,input_shape=(inp_dim,)))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # Compile model
    model.compile(loss=rmse, optimizer=Adam(.001), metrics=['mse'])
    return model

path='iBeacon_RSSI_Labeled.csv'
x = read_csv(path, index_col=None)
x['x'] = x['location'].str[0]
x['y'] = x['location'].str[1:]
x.drop(["location"], axis = 1, inplace = True)
x["x"] = x["x"].apply(fix_pos)
x["y"] = x["y"].astype(int)

y = x.iloc[:, -2:]
x = x.iloc[:, 1:-2]
train_x, val_x, train_y, val_y = train_test_split(x,y, test_size = .2, shuffle = False)

dropouts = [0.0, 0.08, 0.16, 0.24, 0.32] #0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for dropout in dropouts:
    model = create_deep(train_x.shape[1],dropout)
    hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=100, batch_size=100,  verbose=0)
    preds = model.predict(val_x)
    l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
    print('dropout - {}\t l2dists_mean - {}'.format(dropout,l2dists_mean))
    #plt.plot(hist.history['val_loss'])
'''plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()'''
