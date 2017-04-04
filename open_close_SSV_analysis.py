# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:29:22 2017

@author: azfv1n8
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from itertools import combinations
import cPickle as pickle


#%% Data preprocessing

prod = pd.read_pickle('data/results/heat_prod_1979_2016.pkl') # generate by running gen_38y_production.py
price_df = pd.read_pickle('data/results/tot_prices_1979_2016.pkl') # generate by running dev_price_models.apply_price_model(...)
years = range(1979,2017)

def split_spring_fall(split_date=dt.datetime(2016,7,17,1), price_df=price_df, years=years):
    springs = OrderedDict()
    falls = OrderedDict()
    for yr in years:
        split_date = split_date.replace(yr)
        this_year = price_df[price_df.index.year==yr]
        springs[yr] = this_year[this_year.index<split_date]
        falls[yr] = this_year[this_year.index>=split_date]
    
    return springs, falls
    
    
    
def spring_cost(spring_df, close_date, relative=True):
    
    if any(np.isnan(spring_df.ix[spring_df.index>=close_date, 'price_euro_noSSV'])):
        return np.nan
    
    tot_cost = spring_df.ix[spring_df.index<close_date, 'price_euro_wSSV'].sum() \
                + spring_df.ix[spring_df.index>=close_date, 'price_euro_noSSV'].sum()
                
    if relative:
        return tot_cost/spring_df['price_euro_wSSV'].sum()
    elif not relative:
        return tot_cost
        
        
def fall_cost(fall_df, open_date, relative=True):
    if any(np.isnan(fall_df.ix[fall_df.index<open_date, 'price_euro_noSSV'])):
        return np.nan       
    
    tot_cost = fall_df.ix[fall_df.index<open_date, 'price_euro_noSSV'].sum() \
                + fall_df.ix[fall_df.index>=open_date, 'price_euro_wSSV'].sum()
                
    if relative:
        return tot_cost/fall_df['price_euro_wSSV'].sum()
    elif not relative:
        return tot_cost


def get_tot_yrly_cost(close_date, open_date, year, springs, falls):
    
    return spring_cost(springs[year], close_date, relative=False) \
            + fall_cost(falls[year], open_date, relative=False)
            
def get_all_opt_tot_yrly_costs():
    spring_costs, fall_costs = all_spring_fall_cost_sweeps()
    spring_dfs, fall_dfs = split_spring_fall()
    
    
    min_dates = min_dates_boxplots(spring_costs, fall_costs, save=False)
    min_dates.index = years
    plt.close()
    
    res = OrderedDict()
    w_SSV_cost = OrderedDict()
    for yr in years:
        close_date = min_dates.ix[yr, 'Spring']
        open_date = min_dates.ix[yr, 'Fall']
        res[yr] = get_tot_yrly_cost(close_date, open_date, yr, spring_dfs, fall_dfs)
        w_SSV_cost[yr] = spring_dfs[yr]['price_euro_wSSV'].sum() + fall_dfs[yr]['price_euro_wSSV'].sum()
        
    return res, w_SSV_cost

        
def all_spring_fall_cost_sweeps(frequency='D', relative_cost=True, years=years, springs=None, falls=None):      
    if springs is None or falls is None:
        springs, falls = split_spring_fall()
    spring_costs = OrderedDict()
    fall_costs = OrderedDict()
    
    for yr in years:
        print yr
        sd = springs[yr]
        cost_s = pd.Series(index=pd.date_range(sd.index[0], sd.index[-1], freq=frequency))
        for cd in cost_s.index:
            cost_s[cd] = spring_cost(sd, cd, relative_cost)    
        spring_costs[yr] = cost_s
       
        fd = falls[yr]
        cost_f = pd.Series(index=pd.date_range(fd.index[0], fd.index[-1], freq=frequency))
        for cd in cost_f.index:
            cost_f[cd] = fall_cost(fd, cd, relative_cost)
        fall_costs[yr] = cost_f
        
    return spring_costs, fall_costs


def get_pred_dfs(spring_costs, fall_costs):
    """ return dataframes with possible predictor variables for the
        optimal closing time """
           
    min_dates = min_dates_boxplots(spring_costs, fall_costs, save=False)
    min_dates.index = years
    for season in ['Spring', 'Fall']:
        for yr in years:
            min_ts = min_dates.ix[yr, season]
            min_dates.at[yr, '%s_prod' % season] = prod.ix[min_ts]
            min_dates.at[yr, '%s_1wprod' % season] = prod.ix[min_ts+dt.timedelta(days=-7):min_ts].mean()
            min_dates.at[yr, '%s_2wprod' % season] = prod.ix[min_ts+dt.timedelta(days=-14):min_ts].mean()


            
    return min_dates
        
    

#%% plots
   
def plot_oc_costs_all(spring_costs, fall_costs,\
                        ylabel='Total production cost relative to reference', \
                        save_suffix='allyears_cost_vs_octime', relative_cost=True, save=False):
    for tit, data, xlabel in zip(['Spring', 'Fall'], [spring_costs, fall_costs], ['SSV closing date', 'SSV opening date']):
        fig, ax = plt.subplots(1,1, figsize=(20,15))
        colormap=plt.get_cmap('Vega20')
        prop_cycle=(cycler('color', 2*[colormap(c) for c in np.linspace(0,1,len(years)/2)]) +\
                           cycler('linestyle', ['-', '--']*(len(years)/2))) 
        for yr, p in zip(years, prop_cycle):
            ts = [dt.datetime(1904, t.month, t.day, t.hour) for t in data[yr].index]
            ax.plot_date(ts, data[yr], p['linestyle'], label=str(yr), c=p['color'])
            min_ts = data[yr].idxmin()
            ax.plot_date(dt.datetime(1904, min_ts.month, min_ts.day, min_ts.hour), data[yr].min(), 'ko', markerfacecolor=p['color'])
            ax.grid('on')
        if relative_cost:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        elif not relative_cost:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1e6))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
        fmt = mdates.DateFormatter('%b %d')
        ax.xaxis.set_major_formatter(fmt)
        ax.set_ylabel('Produktion [MW]')
        ax.set_title(tit)
        ax.set_ylabel(ylabel)
        ax.legend(ncol=5, loc=4)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        if save:
            fig.savefig('figure/%s_%s.pdf'%(tit, save_suffix))
    
      
def plot_oc_costs_subplots(spring_costs, fall_costs, relative_cost=True, save=False,\
                           save_suffix='relcost_vs_octime_subplt',\
                           ylabel='Total production cost relative to reference'):
    colormap=plt.get_cmap('Set1')

    for tit, data, xlabel in zip(['Spring', 'Fall'], [spring_costs, fall_costs], ['SSV closing date', 'SSV opening date']):
        fig, axes = plt.subplots(3,3, figsize=(15,12), sharex=True, sharey=True)
        
        yrlist = [years[5*i:5*i+5] for i in range(2)] + [years[4*i+2:4*i+6] for i in range(2,9)] # this splits the years time series into 9 chunks
        print yrlist
        for ax, yrs in zip(axes.ravel(), yrlist):
            prop_cycle=cycler('color', [colormap(c) for c in np.linspace(0,1,len(yrs))])
            for yr, p in zip(yrs, prop_cycle):
                ts = [dt.datetime(1904, t.month, t.day, t.hour) for t in data[yr].index]
                ax.plot_date(ts, data[yr], '-', label=str(yr), c=p['color'])
                min_ts = data[yr].idxmin()
                ax.plot_date(dt.datetime(1904, min_ts.month, min_ts.day, min_ts.hour), data[yr].min(), 'ko', markerfacecolor=p['color'])
            ax.legend(loc='best')
            if relative_cost:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
            elif not relative_cost:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1e6))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(28))
            fmt = mdates.DateFormatter('%b %d')
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(axis='y', labelleft='on')
            ax.grid('on', linestyle=':')
            if yrlist.index(yrs)==3:
                ax.set_ylabel(ylabel)
            if yrlist.index(yrs)==7:
                ax.set_xlabel(xlabel)
            
        fig.suptitle(tit)
        fig.tight_layout()
        fig.savefig('figure/%s_%s.pdf'%(tit, save_suffix))
        
        
def min_cost_boxplots(spring_costs, fall_costs, save=False):
    min_relcosts = pd.DataFrame(columns= ['Spring', 'Fall'])

    for c, tit in zip([spring_costs, fall_costs], ['Spring', 'Fall']):
        min_relcosts[tit] = [c[yr].min() for yr in years]

    fig, ax = plt.subplots()
    ax.boxplot(min_relcosts.values, sym='k+', notch=True)
    plt.xticks([1, 2], ['Spring', 'Fall'])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.grid('on', linestyle=':')
    ax.set_ylabel('Total production cost at optimum relative to reference')
    
    if save:
        fig.savefig('figure/min_cost_boxplot.pdf')
    
    return min_relcosts


def min_dates_boxplots(spring_costs, fall_costs, savefig=False, savedates=False):
    min_dates = pd.DataFrame(columns= ['Spring', 'Fall'])
    edge_dates = pd.DataFrame(columns= ['Spring', 'Fall'])
    ytls = [['Earliest safe \nclosing time', 'Optimal SSV \nclosing time'],\
            [ 'Latest safe \nopening time', 'Optimal SSV \nopening time']]

    for c, season, yticklabels in zip([spring_costs, fall_costs], ['Spring', 'Fall'], ytls):
        min_dates[season] = [c[yr].idxmin() for yr in years]
        min_dates[season + '_sameyr'] = [dt.datetime(1972, t.month, t.day, t.hour) for t in min_dates[season]]
        min_dates[season + '_timestamps'] = [to_unix_ts(t) for t in min_dates[season + '_sameyr']]
        
        if season=='Spring':
            edge_dates[season] = [c[yr].dropna().index.min() for yr in years]
        elif season=='Fall':
            edge_dates[season] = [c[yr].dropna().index.max() for yr in years]
        edge_dates[season + '_sameyr'] = [dt.datetime(1972, t.month, t.day, t.hour) for t in edge_dates[season]]
        edge_dates[season + '_timestamps'] = [to_unix_ts(t) for t in edge_dates[season + '_sameyr']]
    
        fig, ax = plt.subplots(figsize=(12,8))
        ax.boxplot([edge_dates[season + '_timestamps'].values, min_dates[season + '_timestamps'].values], sym='k+', notch=True, vert=False)
        plt.yticks([1,2], yticklabels, rotation='vertical')
        xtickpos = [to_unix_ts(min(min_dates[season + '_sameyr'].min(), edge_dates[season + '_sameyr'].min()) + dt.timedelta(days=7*i)) for i in range(-1, 12)]
        xtick_dts = [dt.datetime.utcfromtimestamp(ts) for ts in xtickpos]
        plt.xticks(xtickpos, [t.strftime('%b %d') for t in xtick_dts])
        ax.grid('on', linestyle=':')
        ax.set_title(season)
        
        median = dt.datetime.utcfromtimestamp(min_dates[season + '_timestamps'].median())
        quartile_1 = dt.datetime.utcfromtimestamp(min_dates[season + '_timestamps'].quantile(0.25))
        quartile_3 = dt.datetime.utcfromtimestamp(min_dates[season + '_timestamps'].quantile(0.75))

        print season + ':'
        print '\tMedian: ' + median.strftime('%b %d')
        print '\t1st quartile: ' + quartile_1.strftime('%b %d')
        print '\t3st quartile: ' + quartile_3.strftime('%b %d')
        if savefig:
            fig.savefig('figure/%s_dates_boxplot.pdf' % season)
            
        if savedates:
            min_dates.to_pickle('data/results/mincost_oc_dates.pkl')
    return min_dates
    

def get_percentage_saved_octimes(spring_costs, fall_costs, percentage=0.9):
    spring_dates = OrderedDict()
    fall_dates = OrderedDict()
    for c, season, dates in zip([spring_costs, fall_costs], ['Spring', 'Fall'], [spring_dates, fall_dates]):
        for yr in years:
            min_cost = c[yr].min()
            perc_saved_level = percentage*(min_cost - 1.) + 1.
            dates[yr] = c[yr][c[yr]<=perc_saved_level].index
            
    return spring_dates, fall_dates
 
    
def plot_subopt_octimes(spring_costs, fall_costs, percentage=0.9, height=0.5, save=False):
    spring_dates, fall_dates = get_percentage_saved_octimes(spring_costs, fall_costs, percentage=percentage)
    s_rects = OrderedDict()
    f_rects = OrderedDict()
    
    spring_bunch, fall_bunch = bunch_subopt_octimes(spring_costs, fall_costs, percentage)
    
    for dates, rectangles, season, c, min_date, b_dates, xlabel in zip([spring_dates, fall_dates], \
                                            [s_rects, f_rects], \
                                            ['Spring', 'Fall'], \
                                            [spring_costs, fall_costs],\
                                            [dt.datetime(2017,3,21,1), dt.datetime(2017,9,2,1,1)],\
                                            [spring_bunch, fall_bunch],\
                                            ['SSV closing date', 'SSV opening date']):
        min_dates = []
        print dates
        fig, [ax1, ax2] = plt.subplots(2,1,figsize=(12,12), gridspec_kw={'height_ratios':[2,1]})
        ax1.set_axisbelow(True)
        ax1.grid('on', linestyle=':')
        ax2.grid('on', linestyle=':')

        for yr in years:
            rectangles[yr] = get_oc_rects(dates[yr], spring_costs[yr].index.freqstr)
            for r in rectangles[yr]:
                r.set_y(yr - 0.5*height)
                r.set_height(height)
                ax1.add_patch(r)
            min_dates.append(dates[yr].min())
            
            if season=='Spring':
                eo_date = c[yr].dropna().index.min()
                rect = get_earliest_open_rect(eo_date)
                rect.set_y(yr - 0.5*height)
                rect.set_height(height)
                ax1.add_patch(rect)
                
            elif season=='Fall':
                lc_date = c[yr].dropna().index.max()
                rect = get_last_close_rect(lc_date)
                rect.set_y(yr - 0.5*height)
                rect.set_height(height)
                ax1.add_patch(rect)
            
        ax1.set_title('%s %2.1f %% of total benefit' % (season, 100*percentage))
        xtickpos = [to_unix_ts(same_year(min_date) + dt.timedelta(days=7*i)) for i in range(17)]
        xtick_dts = [dt.datetime.utcfromtimestamp(ts) for ts in xtickpos]
        ax1.xaxis.set_ticks(xtickpos)
        ax1.xaxis.set_ticklabels([t.strftime('%b %d') for t in xtick_dts])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.set_ylim(years[0]-0.5, years[-1]+0.5)
        ax1.set_xlim(xtickpos[0], xtickpos[-1])             

        no_bins = (max(b_dates) - min(b_dates)).days + 1    
        ll=ax2.hist(b_dates, bins=no_bins) 
        print ll
        if season=='Spring':
            eo_dates = pd.Series([same_year(c[yr].dropna().index.min()) for yr in years])
            quantiles = [eo_dates.quantile(q) for q in (1, .85)]
        elif season=='Fall':
            lc_dates = pd.Series([same_year(c[yr].dropna().index.max()) for yr in years])
            quantiles = [lc_dates.quantile(q) for q in (0., .15)]

        ax2.vlines(quantiles[0], 0, len(years), 'r', linestyle='-', label='100% safe')
        ax2.vlines(quantiles[1], 0, len(years), 'r', linestyle='--', label='85% safe')
        ax2.hlines(len(years), *ax2.get_xbound(), color='k', linestyle='--', label='All years')
        ax2.set_ylim(0, len(years)+2)
        ax2.xaxis.set_ticks(xtick_dts)
        ax2.xaxis.set_ticklabels([t.strftime('%b %d') for t in xtick_dts])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Count')
        ax2.set_xlim(min(xtick_dts), max(xtick_dts))

        ax2.legend()

        fig.tight_layout()
        if save:
            fig.savefig('figure/%s_%ipct_benefit.pdf'% (season, 100*percentage))

    return



def bunch_subopt_octimes(spring_costs, fall_costs, percentage=0.9):
    spring_dates, fall_dates = get_percentage_saved_octimes(spring_costs, fall_costs, percentage=percentage)
    spring_bunch, fall_bunch = [], []
    
    for dates, b in zip([spring_dates, fall_dates], [spring_bunch, fall_bunch]):
        for yr in years:
            same_yr_dates = same_year(dates[yr])
            b.extend(same_yr_dates)
        
    return spring_bunch, fall_bunch

#%% auxilary functions for plots with the Rectangle class

def get_oc_rects(dates, freq='D'):
    exp_timestep = pd.Timedelta('1'+freq).total_seconds()
    same_yr_dates = same_year(dates)
    unix_dates = [to_unix_ts(t) for t in same_yr_dates]
    diffs = np.diff(unix_dates)
    x_starts = [unix_dates[0]]
    widths = []
    current_x_start = x_starts[0]
    for i in range(len(unix_dates)-1):
        if diffs[i] > exp_timestep:
            widths.append(unix_dates[i] - current_x_start + exp_timestep)
            current_x_start = unix_dates[i+1]
            x_starts.append(current_x_start)
    
    widths.append(unix_dates[-1] - x_starts[-1] + exp_timestep)
    
    rectangles = []
    for x, w in zip(x_starts, widths):
        rectangles.append(Rectangle((x,0), w, 1.))
    
    return rectangles


def get_earliest_open_rect(eo_date):
    x_end = to_unix_ts(same_year(eo_date))
    x_start = to_unix_ts(same_year(dt.datetime(2017,1,1,0)))
    width = x_end-x_start
    
    return Rectangle((x_start, 0), width, 1., fc='r')


def get_last_close_rect(lc_date):
    x_start = to_unix_ts(same_year(lc_date))
    x_end = to_unix_ts(same_year(dt.datetime(2017,12,31,0)))
    width = x_end-x_start
    
    return Rectangle((x_start, 0), width, 1., fc='r')


def save_hourly_costs():
    spring_costs, fall_costs = all_spring_fall_cost_sweeps(frequency='H')
    
    for season, cost in zip(['Spring', 'Fall'], [spring_costs, fall_costs]):
        path = 'data/results/'
        fname = '%s_costs_h.pkl' % season
        with open(path+fname, 'w') as f:
            pickle.dump(cost, f)
            
    return

#%% time auxilary functions

def same_year(datetimes, year=1972):
    try:
        return [dt.datetime(year, t.month, t.day, t.hour) for t in datetimes]
    except TypeError:
        return dt.datetime(year, datetimes.month, datetimes.day, datetimes.hour)
        
        
def to_unix_ts(datetimes):
    try:
        return [(t - dt.datetime(1970,1,1)).total_seconds() for t in datetimes]
    except TypeError:
        return (datetimes - dt.datetime(1970,1,1)).total_seconds()


#%% production clustering
def reshape_prod(prod):
    prod.at[prod.index[0]+dt.timedelta(hours=-1)] = prod.ix[0]
    df = pd.DataFrame(columns=years, index=range(8760))
    for yr in years:
        this_yr = prod[prod.index.year==yr]
        df[yr] = [this_yr[ts] for ts in this_yr.index if not (ts.month==2 and ts.day==29)]
        
    return df
    
def plot_dist_between_years(reshape_prod):
    new_df = pd.DataFrame(columns=years, index=years)
    for y1, y2 in combinations(years, 2):
        new_df.at[y1,y2] = float(np.sqrt(np.mean((reshape_prod[y1]-reshape_prod[y2])**2)))
        new_df.at[y2,y1] = new_df.ix[y1,y2]
    
    plt.imshow(new_df.values.astype(float))
    plt.colorbar
    
    return new_df

#%%        
def main():
    plt.close('all')
    return
    spring_costs, fall_costs = all_spring_fall_cost_sweeps()
    plot_oc_costs_all(spring_costs, fall_costs, save=True)
    plot_oc_costs_subplots(spring_costs, fall_costs, save=True)
    min_cost_boxplots(spring_costs, fall_costs, save=True)
    min_dates_boxplots(spring_costs, fall_costs, save=True)
        
if __name__=="__main__":
    main()
            