# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 12:52:07 2017

@author: azfv1n8
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from heat_load_model import daily_downsample, hourly_upsample, P_model_erf, two_step_FGLS, print_fitres
from intra_day_profile import apply_idayprof_ww_seasonal
from dev_price_models import apply_price_model
from open_close_SSV_analysis import split_spring_fall, spring_cost, fall_cost, all_spring_fall_cost_sweeps

#%%
# prepare data
full_df = pd.read_pickle('data/input/prod_Tout2014_2016.pkl')  
ts1 = dt.datetime(2014,1,1,1)
ts2 = dt.datetime(2016,1,1,0)

fit_data = full_df.ix[ts1:ts2, :]    
fit_daily_dat = daily_downsample(fit_data) # len 730
fit_daily_means = hourly_upsample(fit_daily_dat) # len 17520
fit_prof = fit_data['prod']/fit_daily_means['prod']

ts3 = ts2 + dt.timedelta(hours=1)
ts4 = full_df.index[-1]
vali_data = full_df.ix[ts3:ts4,:]
vali_daily_dat = daily_downsample(vali_data)


popt, pcov = two_step_FGLS(fit_daily_dat['prod'], fit_daily_dat['Tout'], func=P_model_erf)

vali_daily_mod = P_model_erf(vali_daily_dat['Tout'], *popt)
daily_mod_ups = hourly_upsample(vali_daily_mod.to_frame(name='prod'))

prod_modeled = apply_idayprof_ww_seasonal(daily_mod_ups, fit_prof)

plt.figure()
plt.plot_date(vali_data.index, vali_data['prod'], 'k-')
plt.plot_date(vali_data.index, prod_modeled, 'r-')

err = prod_modeled - vali_data['prod']

print "Brabrand Syd data 2016"
print "RMSE: ", np.sqrt(np.mean(err**2))
print "ME:", np.mean(err)
print "MAE", np.mean(np.abs(err))
print "MApE", np.mean(np.abs(err/vali_data['prod']))

sns.jointplot(vali_data['prod'], prod_modeled, alpha=0.05)


plt.figure()
plt.hist(vali_data['prod'], label='Real prod', bins=50)
plt.hist(prod_modeled, color='r', alpha=0.5, label='Model', bins=50)
plt.title('Brabrand Syd data 2016')
plt.legend()

#%%
atlas_Tout = pd.read_pickle('data/input/Tout1979_2016_56.05_10.0_bias_corrected.pkl')

atlas_daily_2016_Tout = daily_downsample(atlas_Tout.ix[dt.datetime(2016,1,1,1):dt.datetime(2017,1,1,0),:])
atlas_daily_mod = P_model_erf(atlas_daily_2016_Tout, *popt)
atlas_daily_mod = atlas_daily_mod.rename(columns={'Tout':'prod'})
daily_mod_ups_atlas = hourly_upsample(atlas_daily_mod)

prod_modeled_atlas = apply_idayprof_ww_seasonal(daily_mod_ups_atlas, fit_prof)

err_at = prod_modeled_atlas - vali_data['prod']

print "Atlas data 2016"
print "RMSE: ", np.sqrt(np.mean(err_at**2))
print "ME:", np.mean(err_at)
print "MAE", np.mean(np.abs(err_at))
print "MApE", np.mean(np.abs(err_at/vali_data['prod']))


sns.jointplot(vali_data['prod'], prod_modeled_atlas, alpha=0.05)

plt.figure()
plt.hist(vali_data['prod'], label='Real prod', bins=50)
plt.hist(prod_modeled_atlas, color='r', alpha=0.5, label='Model', bins=50)
plt.title('Atlas data 2016')
plt.legend()
#

#%%
years = [2014, 2015, 2016]
prodpath = 'data/input/prod2014_2016.pkl'
full_df['prod'].to_pickle(prodpath)
pricepath = 'data/results/price_vali_2014_2016.pkl'
model_pricepath = 'data/results/tot_prices_1979_2016.pkl'

price = apply_price_model(production_path=prodpath, save=False, savepath=pricepath)
model_price = pd.read_pickle(model_pricepath)

springs, falls = split_spring_fall(price_df=price, years=years)
spring_costs, fall_costs = all_spring_fall_cost_sweeps(years=years, springs=springs, falls=falls)

m_spring_costs, m_fall_costs = all_spring_fall_cost_sweeps()
#%%
ylabel='Total production cost relative to reference'
for yr in years:
    plt.figure()
    plt.plot(spring_costs[yr], 'k-', label='Actual production')
    plt.plot(m_spring_costs[yr], 'r-', label='Modeled production')
    plt.title('Spring %i' % yr)
    plt.ylabel(ylabel)
    plt.xlabel('SSV closing date')
    plt.legend()
    
    plt.figure()
    plt.plot(fall_costs[yr], 'k-', label='Actual production')
    plt.plot(m_fall_costs[yr], 'r-', label='Modeled production')
    plt.title('Fall %i' % yr)
    plt.ylabel(ylabel)
    plt.xlabel('SSV opening date')
    plt.legend()