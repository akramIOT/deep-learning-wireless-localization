from keras.layers import Conv2D, MaxPooling2D, Flatten, Conv2DTranspose
from pandas import read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse

from keras.models import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

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

beacon_coords = {"b3001": (5, 9), 
                 "b3002": (10, 14), 
                 "b3003": (13, 14), 
                 "b3004": (18, 14), 
                 "b3005": (10, 11), 
                 "b3006": (13, 11), 
                 "b3007": (18, 11), 
                 "b3008": (10, 8), 
                 "b3009": (4, 3), 
                 "b3010": (10, 3), 
                 "b3011": (13, 3), 
                 "b3012": (18, 3), 
                 "b3013": (22, 3),}
if __name__ == "__main__":
  # create the top-level parser
  parser = argparse.ArgumentParser(
    description="Submit test to run the nightly .")

  parser.add_argument(
    "--learning_rate",
    default=0.001,
    type=float,
    help="learning_rate.")
  parser.add_argument(
    "--beta1",
    default=0.9,
    type=float,
    help="beta1.")

  # Parse the args
  args = parser.parse_args()
  #path='../data/iBeacon_RSSI_Labeled.csv'
  path='/opt/iBeacon_RSSI_Labeled.csv'
  x = read_csv(path, index_col=None)
  x['x'] = x['location'].str[0]
  x['y'] = x['location'].str[1:]
  x.drop(["location"], axis = 1, inplace = True)
  x["x"] = x["x"].apply(fix_pos)
  x["y"] = x["y"].astype(int)

  y = x.iloc[:, -2:]
  x = x.iloc[:, 1:-2]
  img_x = np.zeros(shape = (x.shape[0], 25, 25, 1, ))
  for key, value in beacon_coords.items():
    img_x[:, value[0], value[1], 0] -= x[key].values/200
    #print(key, value)
    train_x, val_x, train_y, val_y = train_test_split(img_x, y, test_size = .2, shuffle = False)

  inputs = Input(shape=(train_x.shape[1], train_x.shape[2], 1))
# a layer instance is callable on a tensor, and returns a tensor
  a = Conv2D(12, kernel_size=(7,7), activation='relu', padding = "valid", data_format="channels_last")(inputs)
  a = MaxPooling2D(2)(a)
  a = Conv2D(12, kernel_size=(5,5), activation='relu', padding = "valid", data_format="channels_last")(a)
  a = MaxPooling2D(2)(a)
#a = Conv2D(12, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(a)
#a = MaxPooling2D(2)(a)
  a = Dense(24, activation='relu')(Flatten()(a))
  predictions = Dense(2, activation='relu')(a)

# This creates a model that includes
# the Input layer and three Dense layers
  model = Model(inputs=inputs, outputs=predictions)
  model.compile(optimizer=Adam(args.learning_rate,args.beta1),
              loss=rmse,
              metrics=['mse'])
  hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=100, batch_size=20,  verbose=1)
  preds = model.predict(val_x)
  l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
  print('l2_loss={}'.format(l2dists_mean))
  sortedl2_deep = np.sort(l2dists)
  dist_acc = []
  for i in [1, 2, 3, 4, 5]:
    dist_acc = dist_acc + [np.sum(sortedl2_deep <= i)/np.size(sortedl2_deep)]
  print(dist_acc)
