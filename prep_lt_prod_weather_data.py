# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 10:45:58 2017

@author: azfv1n8
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) + '\\dmi_ensemble_handler\\')
import sql_tools as sq
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    try:
        save = argv[0]
        print save
    except:
        save = False
    load_path = 'data/input/'
    
    ts1 = dt.datetime(2014,1,1,1)
    ts2 = dt.datetime(2017,1,1,0)
    
    prod = sq.fetch_production(ts1, ts2)
    Tout = sq.fetch_BrabrandSydWeather('Tout', ts1, ts2)
    
    df = pd.DataFrame(index=pd.date_range(ts1, ts2, freq='H'))
    df['prod'] = prod
    df['Tout'] = Tout
    
    plt.plot_date(df.index, df['Tout'])
   
    cons_df = pd.read_pickle(load_path + 'heat_consumption.pkl')
    
    # clean error from data outage in the temperature:
    df.at[dt.datetime(2014,6,7,3):dt.datetime(2014,6,10,14), 'Tout'] = np.mean((df.ix[dt.datetime(2014,6,7,2), 'Tout'],\
                                                                                    df.ix[dt.datetime(2014,6,10,15), 'Tout']))
    #%% Cleaning up errors in the production time series:
    # This method works, because I have already cleaned and validated the consumption data
    cleaning_log = {}
    for ts in cons_df.index:
        if np.abs(df.ix[ts, 'prod'] - cons_df.ix[ts, 'consumption']) > 80:
            df.at[ts, 'prod'] = cons_df.ix[ts, 'consumption']
            cleaning_log[ts] = {'before':df.ix[ts, 'prod'], 'after':cons_df.ix[ts, 'consumption']}
            
    if save:    
        df.to_pickle(load_path + 'prod_Tout2016_2016.pkl')
    
    return df
    
if __name__=='__main__':
    main(main(sys.argv[1:]))