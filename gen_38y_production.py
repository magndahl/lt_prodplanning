# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:07:56 2017

@author: azfv1n8
"""


import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from heat_load_model import daily_downsample, hourly_upsample, P_model_erf, two_step_FGLS, print_fitres
from intra_day_profile import apply_idayprof_ww_seasonal



# prepare data
full_df = pd.read_pickle('data/input/prod_Tout2014_2016.pkl')  
ts1 = dt.datetime(2014,1,1,1)
ts2 = dt.datetime(2016,1,1,0)

fit_data = full_df.ix[ts1:ts2, :]    
fit_daily_dat = daily_downsample(fit_data) # len 730
fit_daily_means = hourly_upsample(fit_daily_dat) # len 17520
fit_prof = fit_data['prod']/fit_daily_means['prod']

popt, pcov = two_step_FGLS(fit_daily_dat['prod'], fit_daily_dat['Tout'], func=P_model_erf)
print_fitres(popt, pcov)
#%%
atlas_Tout = pd.read_pickle('data/input/Tout1979_2016_56.05_10.0_bias_corrected.pkl')
## Generate 38 years worth of heat load data
atlas_daily_Tout = daily_downsample(atlas_Tout)
atlas_38yr_daily_mod = P_model_erf(atlas_daily_Tout, *popt)
atlas_38yr_daily_mod.rename_axis({'Tout':'prod'}, axis='columns', inplace=True)

atlas_38yr_daily_mod_ups = hourly_upsample(atlas_38yr_daily_mod)
atlas_38yr_mod = apply_idayprof_ww_seasonal(atlas_38yr_daily_mod_ups, fit_prof)

#%%
save=False
if save:
    atlas_38yr_mod.to_pickle('data/results/heat_prod_1979_2016.pkl')