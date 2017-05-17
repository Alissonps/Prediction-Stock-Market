#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#--- Imports
import numpy as np
import pandas as pd
import statsmodels
import warnings
import scipy as sp
import scipy.stats as st
import itertools as it
import statsmodels

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from functools import partial
from multiprocessing import Pool

warnings.filterwarnings("ignore")
def get_aic_bic(order,series):
    aic=np.nan
    bic=np.nan
    #print(series.shape,order)
    try:
        arima_mod=statsmodels.tsa.arima_model.ARIMA(series,order=order,freq='D').fit(transparams=True,method='css', disp=-1)
        aic=arima_mod.aic
        bic=arima_mod.bic
        # print(order,aic,bic)
    except Exception as e:
        # print e
        pass
    return order, aic,bic

warnings.filterwarnings("ignore")
def get_PDQ_parallel(data,n_jobs=4):

    p_val=3
    q_vals=3
    d_vals=2

    best_mse, best_order_selected = float("inf"), None

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(data):
        X_train, X_test = data[train_index], data[test_index]

        pdq_vals=[ (p,q,d) for p in range(p_val) for q in range(q_vals) for d in range(d_vals)]
        get_aic_bic_partial=partial(get_aic_bic,series=X_train)
        p = Pool(n_jobs)
        res=p.map(get_aic_bic_partial, pdq_vals)
        p.close()

        best_aic, best_order = float("inf"), None
        for (order, aic, bic) in res:
            # print order, aic, bic
            if aic < best_aic:
                best_aic = aic
                best_order = order
        try:
            arima_mod = statsmodels.tsa.arima_model.ARIMA(X_test,order=best_order,freq='D').fit(transparams=True, method='css', disp=-1)
            yhat = arima_mod.predict()
            if yhat.shape[0] < X_test.shape[0]:
                X_test = X_test[0:yhat.shape[0]]
            error = mean_squared_error(X_test, yhat)
            if error < best_mse:
                best_mse = error
                best_order_selected = best_order
        except:
            pass

    return best_order_selected


def friedman_test(args):
    """
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / sp.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
    (sp.sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def nemenyi_multitest(ranks):
    """
        Performs a Nemenyi post-hoc test using the pivot quantities obtained by a ranking test.
        Tests the hypothesis that the ranking of each pair of groups are different.

        Parameters
        ----------
        pivots : dictionary_like
            A dictionary with format 'groupname':'pivotal quantity'

        Returns
        ----------
        Comparions : array-like
            Strings identifier of each comparison with format 'group_i vs group_j'
        Z-values : array-like
            The computed Z-value statistic for each comparison.
        p-values : array-like
            The associated p-value from the Z-distribution wich depends on the index of the comparison
        Adjusted p-values : array-like
            The associated adjusted p-values wich can be compared with a significance level

        References
        ----------
        Bonferroni-Dunn: O.J. Dunn, Multiple comparisons among means, Journal of the American Statistical Association 56 (1961) 52–64.
    """
    k = len(ranks)
    values = ranks.values()
    keys = ranks.keys()
    versus = list(it.combinations(range(k), 2))

    comparisons = [keys[vs[0]] + " vs " + keys[vs[1]] for vs in versus]
    z_values = [abs(values[vs[0]] - values[vs[1]]) for vs in versus]
    p_values = [2 * (1 - st.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    m = int(k * (k - 1) / 2.)
    adj_p_values = [min(m * p_value, 1) for p_value in p_values]

    return comparisons, z_values, p_values, adj_p_values

