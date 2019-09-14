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

def create_deep(inp_dim):
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(50, input_dim=inp_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # Compile model
    model.compile(loss=rmse, optimizer=Adam(.001), metrics=['mse'])
    return model

beacon_coords = {"b3001": (5, 9), 
                 "b3002": (9, 14), 
                 "b3003": (13, 14), 
                 "b3004": (18, 14), 
                 "b3005": (9, 11), 
                 "b3006": (13, 11), 
                 "b3007": (18, 11), 
                 "b3008": (9, 8), 
                 "b3009": (2, 3), 
                 "b3010": (9, 3), 
                 "b3011": (13, 3), 
                 "b3012": (18, 3), 
                 "b3013": (22, 3),}

print("Beacon\t\tdataset\t\tmse\t\tl2_dist error")
for a,v in beacon_coords.items():
	path='/opt/iBeacon_RSSI_Labeled.csv'
	x = read_csv(path, index_col=None)
	x[a] = -200
	x = x[(x.iloc[:,-13:].T != -200).any()]
	'''x = x[x[a] == -200]
	x.drop([a], axis = 1, inplace = True)'''
	x['x'] = x['location'].str[0]
	x['y'] = x['location'].str[1:]
	x.drop(["location"], axis = 1, inplace = True)
	x["x"] = x["x"].apply(fix_pos)
	x["y"] = x["y"].astype(int)
	y = x.iloc[:, -2:]
	x = x.iloc[:, 1:-2]/(-200)
	train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = .2, shuffle = False)
	model = create_deep(train_x.shape[1])
	hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=100, batch_size=100,  verbose=0)
	preds = model.predict(val_x)
	l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
	print('{}\t\t{}\t\t{}\t\t{}'.format(a,x.shape[0],hist.history['val_mean_squared_error'][99],l2dists_mean))
