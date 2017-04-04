# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:57:37 2017

@author: azfv1n8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from scipy.interpolate import interp1d


uni_euro = u"\u20AC"

def_col_dict = {'O+VE':'grey', 'RS':'lightgreen', 'O1+2':'green', \
                'O4':'blue', 'BKVV':'yellow', 'BKVV_BP':'yellow', \
                'O1+2_BP':'green', 'O4_BP':'blue', 'SSV':'red', \
                'SSV_OL':'red', 'Other':'#990066', 'VAK_BKVV':'#0099cc',\
                'VAK_SSV':'#cc9900'}
                
def_hatch_dict = {'O+VE':None, 'RS':None, 'O1+2':None, \
                'O4':None, 'BKVV':None, 'BKVV_BP':'X', \
                'O1+2_BP':'X', 'O4_BP':'X', 'SSV':None, 'SSV_OL':'X', \
                'Other':None, 'VAK_BKVV':None, 'VAK_SSV':None}   

no_SSV_cap = 641. # this is the n-1 redundant capacity in the scneario without SSV
w_SSV_cap = 1291. # this is just the full capacity. Not the n-1 redundant, as that would be 540 MW 

def to_euro(DKK):
    return (1/7.43)*DKK
    
def to_DKK(euro):
    return 7.43*euro


def setup_model(P_tot0=400, input_data_path='data/input/input_no_storage.xlsx', include_SSV=True):
    df = pd.read_excel(input_data_path)
    df.index = df['Enhed']
    
    tot_production = 400.
    m = gb.Model()
    
    if include_SSV:
        units = [u for u in df['Enhed']]
    elif not include_SSV:
        units = [u for u in df['Enhed'] if u!='SSV']

    
    cost = {u:df.ix[u, 'Pris [Euro/MWh]'] for u in units}
    SSV_basecost, SSV_varcost = to_euro(np.load('data/results/SSV_costparams.npy'))

    prod_vars = m.addVars(units, name='production', lb=df.ix[units, 'Min. Output [MW]'], ub=df.ix[units, 'Kapacitet [MW]'])
    on_off_vars = m.addVars(units, name='on_off', vtype=gb.GRB.BINARY)
    
    obj_fun = gb.quicksum(gb.QuadExpr(cost[u]*prod_vars[u]*on_off_vars[u]) for u in units if u!='SSV')
    if include_SSV:
        obj_fun.add(gb.QuadExpr(SSV_varcost*prod_vars['SSV']*on_off_vars['SSV'] + SSV_basecost*on_off_vars['SSV']))
        m.addConstr(lhs=on_off_vars['SSV'], sense=gb.GRB.EQUAL, rhs=1)
    m.addQConstr(lhs=gb.quicksum(gb.QuadExpr(prod_vars[u]*on_off_vars[u]) for u in units), \
                            sense=gb.GRB.EQUAL, rhs=tot_production, name='Energy balance')

    m.setObjective(obj_fun, sense=gb.GRB.MINIMIZE)
    m.setParam("OutputFlag",0)
   

    m.update()
    
    return m, units


def solve_model(total_production, model):    
    model.getQConstrs()[0].setAttr('QCRHS', total_production)  
    model.update()
    
    model.optimize()
   
    return model
    

def price_vs_prod(Nsteps, low=40, high=w_SSV_cap, include_SSV=True, return_func=False):
    m, units = setup_model(include_SSV=include_SSV)
    price_per_MWh = np.empty(Nsteps)   
    prod = np.linspace(low, high, Nsteps)
    for p, i in zip(prod, xrange(Nsteps)):
        m = solve_model(p, m)
        price_per_MWh[i] = m.objVal/p
    
    if return_func:
        return interp1d(prod, price_per_MWh, bounds_error=False)
    else:    
        return price_per_MWh, prod
    
    
def prodcomp_vs_prod(Nsteps, low=40, high=w_SSV_cap, include_SSV=True):
    m, units = setup_model(include_SSV=include_SSV)
    prod = np.linspace(low, high, Nsteps)
    index = range(Nsteps)
    df = pd.DataFrame(index=index, columns=units)    
    for p, i in zip(prod, xrange(Nsteps)):
        m = solve_model(p, m)
        onoff_varnames = {u:'on_off[%s]' % u for u in units}
        production_varnames = {u:'production[%s]' % u for u in units}
        for u in units:
            df.at[i, u] = m.getVarByName(onoff_varnames[u]).X*m.getVarByName(production_varnames[u]).X       
    
    df_float_type = df.astype(float)
    
    return df_float_type
    
    
def plot_avg_MWh_price(DKK=True):
    if DKK:
        conv_func = to_DKK
        currency = 'DKK'
    else:
        conv_func = lambda x: x
        currency = uni_euro
        
    no_SSV_price, prod_no_SSV = price_vs_prod(646, low=40, high=no_SSV_cap, include_SSV=False)
    w_SSV_price, prod_w_SSV = price_vs_prod(1186, low=40, high=w_SSV_cap, include_SSV=True)

    plt.figure()
    plt.plot(prod_no_SSV, conv_func(no_SSV_price), 'b-', label="SSV out")
    plt.plot(prod_w_SSV, conv_func(w_SSV_price), 'g-', label="SSV in")
    plt.xlabel('Total production [MW]')
    plt.ylabel('Average heat price [%s/MWh]' % currency)
    plt.legend()
    plt.grid('on')
    plt.savefig('figure/avg_price_vs_prod.pdf')
 

def plot_total_price(DKK=True):
    if DKK:
        conv_func = to_DKK
        currency = 'DKK'
    else:
        conv_func = lambda x: x
        currency = uni_euro
    
    no_SSV_price, prod_no_SSV = price_vs_prod(646, low=40, high=no_SSV_cap, include_SSV=False)
    w_SSV_price, prod_w_SSV = price_vs_prod(1186, low=40, high=w_SSV_cap, include_SSV=True)

    plt.figure()
    plt.plot(prod_no_SSV, conv_func(prod_no_SSV*no_SSV_price), 'b-', label="SSV out")
    plt.plot(prod_w_SSV, conv_func(prod_w_SSV*w_SSV_price), 'g-', label="SSV in")
    plt.xlabel('Total production [MW]')
    plt.ylabel('Total heat price [%s]' % currency)
    plt.title("Total price for 1 hours' production")
    plt.legend()
    plt.grid('on')
    plt.savefig('figure/tot_price_vs_prod.pdf')
    
    
def plot_prod_comp():
    fig, axes = plt.subplots(2, 1, sharey=True, figsize=(10,20))
    
    df_no_SSV = prodcomp_vs_prod(646, low=40, high=no_SSV_cap, include_SSV=False)
    df_w_SSV = prodcomp_vs_prod(1186, low=40, high=w_SSV_cap, include_SSV=True)

    for ax, prod_df in zip(axes, [df_w_SSV, df_no_SSV]):
        cum_df = np.cumsum(prod_df, axis=1)
        prod = prod_df.sum(axis=1)
        for u in prod_df.columns:
            ax.fill_between(prod, cum_df[u], cum_df[u]-prod_df[u], facecolor=def_col_dict[u], hatch=def_hatch_dict[u], label=u)

        ax.legend(loc=2)
        ax.set_xlabel('Total heat load [MW]')
        ax.set_ylabel('Production [MW]')
        ax.set_xlim(0,1300)
        ax.set_ylim(0,1300)

        
    fig.tight_layout()
    fig.savefig('figure/prod_comp.pdf')
    
    
def apply_price_model(production_path='data/results/heat_prod_1979_2016.pkl', save=False, savepath='data/results/tot_prices_1979_2016.pkl'):    
    low = 40
    prod = pd.read_pickle(production_path)
    tot_prices_df = pd.DataFrame(index=prod.index)
    
    for include_SSV, high, suffix in zip([True, False], [w_SSV_cap, no_SSV_cap], ['_wSSV', '_noSSV']):
        Nsteps = int((high - low)*4)
        price_func = price_vs_prod(Nsteps=Nsteps, low=low, high=high, include_SSV=include_SSV, return_func=True)
    
        tot_prices_df['price_euro%s' % suffix] = prod*price_func(prod)

    if save:
        tot_prices_df.to_pickle(savepath)
            
    return tot_prices_df

