# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:48:29 2017

@author: azfv1n8
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import statsmodels.api as sm
from heat_load_model import two_step_FGLS

plt.close('all')

def f(x, a, b):
    return a/x + b

def g(x, b, a):
    return a*x+b

df = pd.read_excel('data/input/SSV_price_prod.xlsx')

var_cost = df['Variabel pris']
price = df['Timens faktiske pris']
prod = df['Varmeproduktion']

plt.figure()
plt.scatter(prod, var_cost)



df_clean = df[np.logical_and(df['Variabel pris']>283, df['Variabel pris']<380)].dropna()


popt, pcov = curve_fit(f, df_clean['Varmeproduktion'], df_clean['Variabel pris'], p0=[3000, 287])

plotx = np.linspace(10, 600, 1000)
plt.plot(plotx, f(plotx, *popt), 'r-')

plt.figure()
plt.scatter(prod, price)


popt2, pcov2 = curve_fit(f, df_clean['Varmeproduktion'], df_clean['Timens faktiske pris'], p0=[15000, 287])
plt.plot(plotx, f(plotx, *popt2), 'r-')
#
popt3, pcov3 = two_step_FGLS(df_clean['Timens faktiske pris'], df_clean['Varmeproduktion'], func=f, p0=[15000, 287]) # this fit uses feasible least square to account fore heteroscedacity
plt.plot(plotx, f(plotx, *popt3), 'g-')

save=False
if save:
    np.save('data/results/SSV_costparams.npy', popt3)
    
    
eta_el = pd.DataFrame(data={'heat_prod':[50,100,150,200,250,300,350,400,450,484],
                               'eta_el':[.271, .314, .361, .385, .4, .407, .414, .418, .421, .421]})


popt4, pcov4 = curve_fit(f, eta_el['heat_prod'], 1/eta_el['eta_el'], p0=[300, 2.4])
X_eta = sm.add_constant(eta_el['heat_prod'])
y_eta = eta_el['heat_prod']/eta_el['eta_el']
res_eta = sm.OLS(y_eta, X_eta).fit()
print res_eta.summary()

plt.figure()
plt.plot(eta_el['heat_prod'], eta_el['heat_prod']/eta_el['eta_el'],'.')
plt.plot(plotx, res_eta.predict(sm.add_constant(plotx)), 'r-')
plt.xlabel(r'$P_q$')
plt.ylabel(r'$P_q/\eta_e$')




X = df_clean['Varmeproduktion']
X = sm.add_constant(X)

y = df_clean['Varmeproduktion']*df_clean['Timens faktiske pris']
res = sm.OLS(y,X).fit()
print res.summary()

popt5, pcov5 = two_step_FGLS(df_clean['Varmeproduktion']*df_clean['Timens faktiske pris'], df_clean['Varmeproduktion'], func=g, p0=[15000, 287])

plt.figure()
plt.plot(df_clean['Varmeproduktion'], y, '.', alpha=0.25)
plt.plot(plotx, res.predict(sm.add_constant(plotx)), 'r-', label='OLS')
plt.plot(plotx, g(plotx, *popt5), 'g-', label='FGLS')
plt.legend()
plt.ylabel('Total cost of the hour [DKK]')
plt.ylabel('Production [DKK]')


save=False
if save:
    np.save('data/results/SSV_costparams.npy', popt5)