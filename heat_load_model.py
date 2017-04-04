# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 12:27:48 2017

@author: azfv1n8
"""

import datetime as dt
import pandas as pd
import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit
import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.close('all')

def piecewise_linear(x, x0, y0, k1, k2):

    return np.piecewise(x, [x < x0, x>=x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    
# This model for the production as a function of temperature or effective temperature
# assumes a superposition of piecewise linear
# functions with the position of the bend being 
# normally distributed read 'production_model.pdf' for documentation    
def P_model_erf(T, P0, B, T0, sigma):
    return B*(T-T0)*0.5*erfc((T-T0)/(np.sqrt(2)*sigma)) \
             - B*sigma/(np.sqrt(2*np.pi))*np.exp(-(T-T0)**2/(2*sigma**2))+P0

def daily_downsample(df):
    """ This method returns daily means of a dataframe
        with right-labeled time series data. Daily means
        correspond to values withing the date of the returned
        DataFrames' index. """
        
    new_df = df.copy()
    new_df.index = df.index + dt.timedelta(hours=-1)   
    
    return new_df.resample('D', label='left', closed='left').mean()
    
    
def hourly_upsample(df):
    index = pd.date_range(df.index[0] + dt.timedelta(hours=1), df.index[-1] + dt.timedelta(hours=24), freq='H')
    ups_df = pd.DataFrame(columns=df.columns, index=index)
    
    Ncols = len(df.columns)

    for d in df.index:
        ups_df.at[d + dt.timedelta(hours=1):d+dt.timedelta(hours=24),:] \
                            = np.tile(np.reshape(df.ix[d], (1,Ncols)), 24)\
                                      .reshape(24, Ncols)
                                      
    return ups_df
    
#%%
def two_step_FGLS(y, x, func=P_model_erf, p0=[120,-35,15,2], plot_reg=False):
    # Step 1:
    popt, pcov = curve_fit(func, x, y, p0=p0)
    s1_err = func(x, *popt) - y
    # variance model
    logu2 = np.log(s1_err**2)
    X = sm.add_constant(x)
    est = sm.OLS(logu2, X).fit()
    sigma = np.sqrt(np.exp(est.predict(X)))
    
    # weigthed leas square fit, weight are 1/sigma^2
    popt_w, pcov_w = curve_fit(func, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
    
    if plot_reg:
        plt.figure()
        plt.plot(x, logu2, '.')
        plt.plot(x, est.predict(X), 'r-')
        plt.figure()
        plt.plot(x, s1_err,'.')
    
    return popt_w, pcov_w
    

def print_fitres(popt, pcov, names=('P0', 'B', 'T0', 'sigma')):
    perr = np.sqrt(np.diag(pcov))    
    popt_err = [t for t in zip(names, tuple(popt), tuple(perr))]
    print popt_err
    
    print ''.join([u'%s = %3.2f \u00B1 %1.2f \n' % t for t in popt_err])
    
    #%%

def main():
    
    full_df = pd.read_pickle('data/input/prod_Tout2014_2016.pkl')  
    ts1 = dt.datetime(2014,1,1,1)
    ts2 = dt.datetime(2016,1,1,0)
    fit_data = full_df.ix[ts1:ts2, :]
    
    plt.figure()
    plt.scatter(fit_data['Tout'], fit_data['prod'])
    
    daily_dat = daily_downsample(fit_data)
    daily_dat['week'] = [d.week for d in daily_dat.index]
    daily_means = hourly_upsample(daily_dat)
    
    
    plt.figure()
    plt.plot_date(fit_data.index, fit_data['prod'], 'k-')
    plt.plot_date(fit_data.index, daily_means['prod'], 'r-')
    plt.plot_date(daily_dat.index, daily_dat['prod'], 'x')
    
    
    popt, pcov = curve_fit(P_model_erf, daily_dat['Tout'], daily_dat['prod'], p0=[120,-35,15,2])
    
    plt.figure()
    plt.scatter(daily_dat['Tout'], daily_dat['prod'], c=[i.week for i in daily_dat.index])
    
    plot_Tout = np.linspace(-5, 25, 200)
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *popt), 'k-')
    plt.colorbar()
    
    #%% try spring/fall models:
    spring_daily = daily_dat[daily_dat['week']<=26]
    fall_daily = daily_dat[daily_dat['week']>26]
    
    spopt, spcov = curve_fit(P_model_erf, spring_daily['Tout'], spring_daily['prod'], p0=[120,-35,15,2])
    fpopt, fpcov = curve_fit(P_model_erf, fall_daily['Tout'], fall_daily['prod'], p0=[120,-35,15,2])
    
    lpopt, lpcov = curve_fit(piecewise_linear, np.array(daily_dat['Tout']), np.array(daily_dat['prod']), p0=[15,140,-35,-.1])
    
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *spopt), 'g-')
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *fpopt), 'r-')
    
    plt.figure()
    plt.scatter(daily_dat['Tout'], daily_dat['prod'], c=[i.week for i in daily_dat.index])
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *popt), 'k-', lw=2)
    plt.plot(plot_Tout, piecewise_linear(plot_Tout, *lpopt), 'r-', lw=2)
    
    
    err = P_model_erf(daily_dat['Tout'], *popt) - daily_dat['prod']
    lerr = piecewise_linear(np.array(daily_dat['Tout']), *lpopt) - daily_dat['prod']
    serr = P_model_erf(daily_dat['Tout'], *spopt) - daily_dat['prod']
    ferr = P_model_erf(daily_dat['Tout'], *fpopt) - daily_dat['prod']


    plt.figure()
    plt.plot(err, 'k-', label='all')
    plt.plot(lerr, label='lin')
    plt.plot(ferr, 'r-', label='fall')
    plt.plot(serr, 'g-', label='spring')
    plt.legend()
    plt.title('Seasonal models?')
    
    popt_w, pcov_w = two_step_FGLS(daily_dat['prod'], daily_dat['Tout'], func=P_model_erf, p0=[120,-35,15,2])
    plt.figure()
    plt.scatter(daily_dat['Tout'], daily_dat['prod'], c=[i.week for i in daily_dat.index])
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *popt), 'k-', lw=2, label='Unweighted LS')
    plt.plot(plot_Tout, P_model_erf(plot_Tout, *popt_w), 'r-', lw=2, label='FGLS')    
    plt.legend()
    
    two_step_FGLS(daily_dat['prod'], daily_dat['Tout'], func=P_model_erf, p0=[120,-35,15,2], plot_reg=True)
    
    



    
if __name__=="__main__":
    main()
