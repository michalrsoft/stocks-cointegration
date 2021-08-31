import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
import seaborn;

pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data as pdr
import datetime
import yfinance as yf

class CointegrationCalculator: 

    def __init__(self): 
        yf.pdr_override()
        seaborn.set(style='whitegrid')

    def __zscore(self, series): 
        return (series - np.mean(series)) / (np.std(series))

    def stationary_adfuller_test(self, x): 
        pvalue = adfuller(x)[1]
        return pvalue

    def get_ratio(self, series1, series2): 
        ratio = series1 / series2
        return self.__zscore(ratio)

    def find_cointegrated_pairs(self, data): 
        n = data.shape[1]

        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))

        keys = data.keys()
        pairs = []
        for i in range(n): 
            for j in range(i + 1, n): 
                S1 = data[keys[i]]
                S2 = data[keys[j]]

                S1.dropna()
                S2.dropna()

                score, pvalue, _ = coint(S1, S2)
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05: 
                    pairs.append((keys[i], keys[j]))

        return score_matrix, pvalue_matrix, pairs