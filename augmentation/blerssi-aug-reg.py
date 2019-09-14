from pandas import read_csv
from pandas import DataFrame
import numpy as np
import pandas as pd


path='../data/iBeacon_RSSI_Labeled.csv'
a = read_csv(path, index_col=None)
#a.drop(["date"], axis = 1, inplace = True)
x_aug = a.groupby('location').filter(lambda b : (len(b) < 10) & (len(b) > 1))
x_aug = x_aug.reset_index(drop=True)
x_aug_1 = x_aug.drop(['location','date'], axis=1)

frames = {}
for loc, df_loc in x_aug.groupby('location'):
    loc_1 = df_loc.drop(['location','date'], axis=1)
    loc_1[loc_1 < -180] = -200
    loc_1 = loc_1.replace(-200,np.NaN)
    frames[loc] = (pd.concat([loc_1.min(), loc_1.max()], axis=1).fillna(-200)).to_dict()

beacon_list = x_aug_1.columns.to_list()

for i, row in df.iterrows():
    location = x_aug.iloc[i]['location']
    for j in beacon_list:
        if frames[location][0][j] != frames[location][1][j]:
            x_aug.set_value(i, j, np.random.randint(frames[location][0][j],frames[location][1][j]))

x_aug.to_csv('../data/iBeacon_RSSI_Labeled_aug_reg.csv',index = False)

