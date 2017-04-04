# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:45:08 2017

@author: azfv1n8
"""
from operator import itemgetter
from collections import OrderedDict
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cross_validation import KFold
from open_close_SSV_analysis import all_spring_fall_cost_sweeps, same_year, to_unix_ts


years = range(1979, 2014)
prod = pd.read_pickle('data/results/heat_prod_1979_2016.pkl') # generate by running gen_38y_production.py

#min_dates = pd.read_pickle('data/results/mincost_oc_dates.pkl')

#with open('data/results/Spring_costs_h.pkl', 'r') as f:
#    spring_costs = pickle.load(f)
#with open('data/results/Fall_costs_h.pkl', 'r') as f:
#    fall_costs = pickle.load(f)

def KF_yrs(n_folds=7, years=years):
    
    train_vali_yrs = OrderedDict()
    for ix, (train_ix, vali_ix) in enumerate(KFold(len(years), n_folds)):
        train_vali_yrs[ix] = {'train_years':list(itemgetter(*train_ix)(years)),
                              'vali_years':list(itemgetter(*vali_ix)(years))}
        
    return train_vali_yrs


def score(yr, oc_date, season, spring_costs=spring_costs, fall_costs=fall_costs):
    if season=='Spring':
        max_saved = 1. - spring_costs[yr].min()
        saved = 1. - spring_costs[yr].ix[oc_date]
    elif season=='Fall':
        max_saved = 1. - fall_costs[yr].min()
        saved = 1. - fall_costs[yr].ix[oc_date]
    
    return saved/max_saved


def get_min_dates(spring_costs=spring_costs, fall_costs=fall_costs):
    years = spring_costs.keys()
    min_dates = pd.DataFrame(index=years, columns=['Spring', 'Fall'])
    
    for season, cost in zip(min_dates.columns, [spring_costs, fall_costs]):
        for yr in years:
            min_dates.at[yr, season] = cost[yr].idxmin()
    
    return min_dates


def fixed_date(md_list, how='median'):
    same_yr = same_year(md_list)
    
    if how=='median':
        return dt.datetime.utcfromtimestamp(np.median(to_unix_ts(same_yr)))
    elif how=='mean':
        return dt.datetime.utcfromtimestamp(np.mean(to_unix_ts(same_yr)))
    else:
        return None
    
    
def cross_vali_fixed_date(season, n_folds=7, years=years, how='median',\
                          spring_costs=spring_costs, fall_costs=fall_costs):
    train_vali_yrs = KF_yrs(n_folds, years)
    
    train_scores = pd.DataFrame(index=range(len(train_vali_yrs[0]['train_years'])), columns=['Round_%i'% i for i in train_vali_yrs.keys()])
    vali_scores = pd.DataFrame(index=range(len(train_vali_yrs[0]['vali_years'])), columns=['Round_%i'% i for i in train_vali_yrs.keys()])
    
    min_dates = get_min_dates(spring_costs, fall_costs)
    
    for vali_round in train_vali_yrs.keys():
        oc_date = fixed_date(min_dates.ix[train_vali_yrs[vali_round]['train_years']][season], how=how)
        for ix, train_yr in enumerate(train_vali_yrs[vali_round]['train_years']):
            train_scores.at[ix, 'Round_%i'%vali_round] = score(train_yr, same_year(oc_date, year=train_yr), season=season)
        for ix, vali_yr in enumerate(train_vali_yrs[vali_round]['vali_years']):
            vali_scores.at[ix, 'Round_%i'%vali_round] = score(vali_yr, same_year(oc_date, year=vali_yr), season=season)
            
    
    return train_scores, vali_scores
        