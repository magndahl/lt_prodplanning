# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:15:38 2017

@author: azfv1n8
"""

import datetime as dt
import pandas as pd
import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit
import calendar

import matplotlib.pyplot as plt
import seaborn as sns
from heat_load_model import daily_downsample, hourly_upsample, P_model_erf, two_step_FGLS



    
    
def mean_profile(prof):
    profile = []
    for i in range(24):
        profile.append(prof.where(prof.index.hour==i).mean())        
        
    return profile
    
def workday_profile(prof):
    profile = []
    for i in range(24):
        profile.append(prof.where(np.logical_and(prof.index.dayofweek<5, prof.index.hour==i)).mean())        
        
    return profile
    
    
def workday_profile_std(prof):
    profile = []
    for i in range(24):
        profile.append(prof.where(np.logical_and(prof.index.dayofweek<5, prof.index.hour==i)).std())        
        
    return profile
    

def weekend_profile(prof):
    profile = []
    for i in range(24):
        profile.append(prof.where(np.logical_and(prof.index.dayofweek>=5, prof.index.hour==i)).mean())        
        
    return profile
    
       
def workday_profile_by_month(prof, month):
    profile = []
    for i in range(24):
        profile.append(prof.where(np.logical_and(prof.index.month==month, \
        np.logical_and(prof.index.dayofweek<5, prof.index.hour==i))).mean())        
        
    return profile
    

def workday_profile_by_season(prof, season):
    seasons_dict = {'winter':(12,1,2), 
                    'spring':(3,4,5),
                    'summer':(6,7,8),
                    'fall':(9,10,11)}
    profiles = []
    for month in seasons_dict[season]:
        profiles.append(workday_profile_by_month(prof, month))
        
    profile = np.array(profiles).mean(axis=0)
        
    return profile
    
    
def weekend_profile_by_season(prof, season):
    seasons_dict = {'winter':(12,1,2), 
                    'spring':(3,4,5),
                    'summer':(6,7,8),
                    'fall':(9,10,11)}
    profiles = []
    for month in seasons_dict[season]:
        profiles.append(weekend_profile_by_month(prof, month))
        
    profile = np.array(profiles).mean(axis=0)
        
    return profile

    
def weekend_profile_by_month(prof, month):
    profile = []
    for i in range(24):
        profile.append(prof.where(np.logical_and(prof.index.month==month, \
        np.logical_and(prof.index.dayofweek>=5, prof.index.hour==i))).mean())        
        
    return profile
    
    
def plot_monthly_workday(prof):    
    plt.figure()
    profiles = []
    cmap = plt.get_cmap('hsv')
    for i in range(1,13):
        c = cmap(float(i)/12)
        profile = workday_profile_by_month(prof, i)
        profiles.append(profile)
        plt.plot(range(1,25), profile,  '-', c=c, label=calendar.month_name[i])
    
    plt.plot(range(1,25), workday_profile(prof), 'k-', lw=2, label='Mean workday profile')


    plt.legend()
    plt.title('Workday profiles')
    plt.xlabel('Hour of day')
    plt.ylabel('Weight')
    
    return profiles
     
       
def plot_monthly_weekend(prof):  
    plt.figure()
    profiles = []
    cmap = plt.get_cmap('hsv')
    for i in range(1,13):
        c = cmap(float(i)/12)
        profile = weekend_profile_by_month(prof, i)
        profiles.append(profile)
        plt.plot(range(1,25), profile,  '-', c=c, label=calendar.month_name[i])

    plt.plot(range(1,25), weekend_profile(prof), 'k-', lw=2, label='Mean weekend profile')
    plt.legend()
    plt.title('Weekend profiles')
    plt.xlabel('Hour of day')
    plt.ylabel('Weight')
    
    return profiles


def plot_seasonal_workday(prof):    
    plt.figure()
    seasons = ['winter', 'spring', 'summer', 'fall']
    profiles = []
    cmap = plt.get_cmap('hsv')
    for s in seasons:
        c = cmap(float(seasons.index(s))/4)
        profile = workday_profile_by_season(prof, s)
        profiles.append(profile)
        plt.plot(range(1,25), profile,  '-', c=c, label=s)
    
    plt.plot(range(1,25), workday_profile(prof), 'k-', lw=2, label='Mean workday profile')

    plt.legend()
    plt.title('Workday profiles')
    plt.xlabel('Hour of day')
    plt.ylabel('Weight')
    
    return profiles
    

def plot_seasonal_weekend(prof):    
    plt.figure()
    seasons = ['winter', 'spring', 'summer', 'fall']
    profiles = []
    cmap = plt.get_cmap('hsv')
    for s in seasons:
        c = cmap(float(seasons.index(s))/4)
        profile = weekend_profile_by_season(prof, s)
        profiles.append(profile)
        plt.plot(range(1,25), profile,  '-', c=c, label=s)
    
    plt.plot(range(1,25), weekend_profile(prof), 'k-', lw=2, label='Mean workday profile')

    plt.legend()
    plt.title('Weekend profiles')
    plt.xlabel('Hour of day')
    plt.ylabel('Weight')
    
    return profiles    


def season_from_month(month_number):
    if month_number in (12,1,2):
        return 'winter'
    elif month_number in (3,4,5):
        return 'spring'
    elif month_number in (6,7,8):
        return 'summer'
    elif month_number in (9,10,11):
        return 'fall'
    else:
        raise LookupError('Specified month must be and integer 1-12. Tried %i' % month_number)
        

def apply_idayprof_ww_seasonal(daily_means, prof):
    prod_wprofile = pd.Series(index=daily_means.index)
    prof_dict = {}
    for season in ('winter', 'spring', 'summer', 'fall'):
        prof_dict[season] = {}
        for daytype in ('workday','weekend'):
            if daytype == 'workday':
                profile = workday_profile_by_season(prof, season)
            elif daytype == 'weekend':
                profile = weekend_profile_by_season(prof, season)
            prof_dict[season][daytype] = profile
            
    
    for ts in prod_wprofile.index:
        season = season_from_month(ts.month)
        # weekday
        if ts.dayofweek < 5:
            profile = prof_dict[season]['workday']
        elif ts.dayofweek >= 5:
            profile = prof_dict[season]['weekend']
        # this is because the production is time stamped on the right interval so hour 0 is the last in the day
        weight = profile[ts.hour]

        prod_wprofile[ts] = weight*daily_means.ix[ts, 'prod']
               
    return prod_wprofile
                      
            
def apply_idayprof_ww_monthly(daily_means, prof):
    prod_wprofile = pd.Series(index=daily_means.index)
    prof_dict = {}
    for month_no in range(1,13):
        prof_dict[month_no] = {}
        for daytype in ('workday','weekend'):
            if daytype == 'workday':
                profile = workday_profile_by_month(prof, month_no)
            elif daytype == 'weekend':
                profile = weekend_profile_by_month(prof, month_no)
            prof_dict[month_no][daytype] = profile
            
    
    for ts in prod_wprofile.index:
        # weekday
        if ts.dayofweek < 5:
            profile = prof_dict[ts.month]['workday']
        elif ts.dayofweek >= 5:
            profile = prof_dict[ts.month]['weekend']
        # this is because the production is time stamped on the right interval so hour 0 is the last in the day
        weight = profile[ts.hour]

        prod_wprofile[ts] = weight*daily_means.ix[ts, 'prod']
               
    return prod_wprofile
            

#%%   
    
def main():
    full_df = pd.read_pickle('data/input/prod_Tout2014_2016.pkl')  
    ts1 = dt.datetime(2014,1,1,1)
    ts2 = dt.datetime(2016,1,1,0)
    fit_data = full_df.ix[ts1:ts2, :]    
       
    daily_dat = daily_downsample(fit_data) # len 730
    daily_means = hourly_upsample(daily_dat) # len 17520
    
    prof = fit_data['prod']/daily_means['prod']
    
    plt.figure()
    plt.plot_date(fit_data.index, prof, '-')
    
    plot_monthly_workday(prof)
    plot_monthly_weekend(prof)
    plot_seasonal_workday(prof)
    plot_seasonal_weekend(prof)
    
    
    prod_wprofile = apply_idayprof_ww_seasonal(daily_means, prof)
    print prod_wprofile
    plt.figure()
    plt.plot_date(fit_data.index, fit_data['prod'], 'k-')
    plt.plot_date(prod_wprofile.index, prod_wprofile, 'r-')
    
    err = prod_wprofile - fit_data['prod']
    print "RMSE: ", np.sqrt(np.mean(err**2))
    print "ME:", np.mean(err)
    print "MAE", np.mean(np.abs(err))
    print "MApE", np.mean(np.abs(err/fit_data['prod']))
    
    
    popt, pcov = two_step_FGLS(daily_dat['prod'], daily_dat['Tout'], p0=[120,-35,15,2])
    print popt, pcov
    plt.figure()
    plt.scatter(daily_dat['Tout'], daily_dat['prod'], c=[i.week for i in daily_dat.index])
    
    plot_Tout = np.linspace(-5, 25, 200)
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *popt), 'k-', lw=2)
    plt.colorbar()
    
    daily_mod = P_model_erf(daily_dat['Tout'], *popt)
    daily_mod_ups = hourly_upsample(daily_mod.to_frame(name='prod'))
    
    
    mod_prod = apply_idayprof_ww_seasonal(daily_mod_ups, prof)
    
    plt.figure()
    plt.plot_date(fit_data.index, fit_data['prod'], 'k-')
    plt.plot_date(mod_prod.index, mod_prod, 'r-')
    
    err2 = mod_prod - fit_data['prod']
    print "\nRMSE: ", np.sqrt(np.mean(err2**2))
    print "ME:", np.mean(err2)
    print "MAE", np.mean(np.abs(err2))
    print "MApE", np.mean(np.abs(err2/fit_data['prod']))


    mod_prod_monthly = apply_idayprof_ww_monthly(daily_mod_ups, prof)
    err3 = mod_prod_monthly - fit_data['prod']
    print "\nRMSE: ", np.sqrt(np.mean(err3**2))
    print "ME:", np.mean(err3)
    print "MAE", np.mean(np.abs(err3))
    print "MApE", np.mean(np.abs(err3/fit_data['prod']))
    plt.plot_date(mod_prod.index, mod_prod_monthly, 'g-')
    
    
    sns.jointplot(fit_data['prod'], mod_prod, alpha=0.05)
    
    sns.jointplot(fit_data['prod'], mod_prod_monthly, alpha=0.05)

    plt.figure()
    plt.hist(fit_data['prod'], label='Real prod', bins=50)
    plt.hist(mod_prod, color='r', alpha=0.5, label='Model', bins=50)
    plt.legend()
    

    
if __name__=="__main__":
    main()
    

    
    
