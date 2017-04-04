# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:23:01 2017

@author: azfv1n8
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def rmse(err):
    return np.sqrt(np.mean(err**2))

load_path = 'data/input/Aarhus_tmp2m_1979_2010/'
load_path_new = 'data/input/Aarhus_tmp2m_2015_2016/'

lats = np.load(load_path + 'latitudes.npy')
longs = np.load(load_path + 'longitudes.npy')
dates = np.load(load_path + 'dates.npy')
temps = np.load(load_path + 'temperature.npy')

lats_new = np.load(load_path_new + 'latitudes.npy')
longs_new = np.load(load_path_new + 'longitudes.npy')
dates_new = np.load(load_path_new + 'dates.npy')
temps_new = np.load(load_path_new + 'temperature.npy')

absolute_zero = -273.15 # degree Celcius

temp_df_old = pd.DataFrame(index=[dt.datetime.fromtimestamp(t) for t in dates])

k=1
for i in range(temps.shape[1]):
    for j in range(temps.shape[2]):
        col_name = 'Tout_pt%i' % k 
        print col_name, lats[i,j], longs[i,j]
        temp_df_old[col_name] = temps[:,i,j] + absolute_zero
        
        k += 1

temp_df_new = pd.DataFrame(index=[dt.datetime.fromtimestamp(t) for t in dates_new])
k=1
for i in range(temps_new.shape[1]):
    for j in range(temps_new.shape[2]):
        col_name = 'Tout_pt%i' % k 
        print col_name, lats_new[i,j], longs_new[i,j]
        temp_df_new[col_name] = temps_new[:,i,j] + absolute_zero
        
        k += 1
        
temp_df = pd.concat((temp_df_old, temp_df_new))



measured_Tout = pd.read_pickle('data/input/prod_Tout2014_2016.pkl')  

plt.figure()
for c in temp_df.columns:
    plt.plot_date(temp_df.index, temp_df[c], '-', label=c)

plt.legend()

#%%
plt.figure()
for c in temp_df.columns:
    diff = temp_df[c] - measured_Tout['Tout']
    diff = diff.dropna()
    diff = diff - diff.mean()
    plt.plot_date(diff.index, diff, '-', label=c)
    plt.legend()
    
    print c, rmse(diff), diff.mean()
    
#%%
overlap_temp_df = temp_df.ix[measured_Tout.index].dropna()
overlap_measured = measured_Tout.ix[overlap_temp_df.index]

print overlap_measured['Tout'].describe()
print overlap_temp_df.describe()

##%% determine bias for point 1: lat: 56.04506774, long 9.9999861:
bias = overlap_temp_df['Tout_pt1'].mean() - overlap_measured['Tout'].mean()

np.save('data/input/atlas_56.05_10.0_Tout_BIAS.npy', bias)

# save bias corrected Tout time series for 56.04506774, long 9.9999861:

save_df = pd.DataFrame(columns=['Tout'], index=temp_df.index)
save_df['Tout'] = temp_df['Tout_pt1'] - bias
save_df.to_pickle('data/input/Tout1979_2016_56.05_10.0_bias_corrected.pkl')