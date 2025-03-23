import pandas as pd
import numpy as np

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

class RBAPair:
    def adf_stationarity_check(self, price:pd.Series, cutoff:float=0.05) -> tuple[float, bool]:
        # H0:= unit root exists (non-stationary)
        p = adfuller(price)[1]
        return (p, p < cutoff)
    
    def linear_reg(self, x2:pd.Series, x1:pd.Series) -> list[float]:
        """
        args: Dependent, Independent
        
        returns: [constant, gradient]
        """
        x1 = sm.add_constant(x1)
        results = sm.OLS(x2, x1).fit()
        x1 = x1.drop(columns=['const'])
        x1 = x1.iloc[:, 0]
        return results.params
    
    def explore_stationarity(self, s1:pd.Series, s2:pd.Series, cutoff:float=0.05) -> pd.DataFrame:
        """
        args: Dependent, Independent, Confidence Score
        
        returns: Dataframe of Results
        """
        #Normal Cointegration
        c,m = self.linear_reg(s1, s2)
        residuals = s1 - m*s2
        p, is_stationary = self.adf_stationarity_check(residuals, cutoff)
        coint_dict = {'simple': (p, is_stationary, m)}
        
        #Ratio
        ratio = s1/s2
        p, is_stationary = self.adf_stationarity_check(ratio, cutoff)
        coint_dict['ratio'] = (p, is_stationary, m) #m is same as normal coinegration

        #Log Transformation
        s1_ln, s2_ln = np.log(s1), np.log(s2)
        c,m = self.linear_reg(s1_ln, s2_ln)
        residuals = s1_ln - m*s2_ln
        p, is_stationary = self.adf_stationarity_check(residuals, cutoff)
        coint_dict['log'] = (p, is_stationary, m)

        coint_df = pd.DataFrame.from_dict(coint_dict, orient='index', columns=['p', 'is_stationary', 'hedge_ratio'])
        return coint_df
    
    def calculate_hurst(self, time_series:np.array, min_lag:int=10, max_lag:int=100, points:int=20) -> tuple[float, float, np.array, np.array]:
        def calculate_rs(time_series:np.array, lag:int) -> float|None:
            n_chunks = len(time_series)//lag
            if n_chunks == 0:
                return None
            
            rs_values = []
            for i in range(n_chunks):
                start, end = i*lag, (i+1)*lag
                chunk = time_series[start:end]
                chunk_mean = np.mean(chunk)
                mean_adj = chunk - chunk_mean
                cum_deviate = np.cumsum(mean_adj)
                r = np.max(cum_deviate) - np.min(cum_deviate)
                s = np.std(chunk)
                
                if s > 0:
                    rs = r/s
                    rs_values.append(rs)
                    
            if len(rs_values) >0:
                return np.mean(rs_values)
            else:
                return None
        
        series_len = len(time_series)
        log_spacing = np.logspace(np.log10(min_lag),
                                np.log10(min(max_lag, series_len//2)),
                                points,
                                dtype=int) #Log Base 10, because min_lag, max_lag is of base 10, ln will not work.
        lags = list(log_spacing[log_spacing>1]) #Lag >= 2 to capture meaningful trends instead of being dominated solely by noise
        rs_values = []
        for lag in lags:
            rs = calculate_rs(time_series, lag)
            if rs is not None:
                rs_values.append(rs)
            else:
                lags.remove(lag)
        
        lags = np.array(lags)
        rs_values = np.array(rs_values)
        ln_lags = np.log(lags)
        ln_rs_values = np.log(rs_values)
        
        coeffs = np.polyfit(ln_lags,ln_rs_values, 1)
        hurst_exponent, c = coeffs
        return hurst_exponent, c, ln_lags, ln_rs_values
    
    def calculate_half_life(self, spread_series:np.array, kappa:float, time:int=1) -> float:
        """
        time refers to:
        Daily - 1 day
        Weekly - 7 days
        Monthly - 30 days
        Yearly - 365 days
        """
        
        def estimate_kappa(spread_series:np.array) -> float:
            dS = np.diff(spread_series)
            mean = np.mean(spread_series)
            x = mean - spread_series[:-1]
            c, kappa = self.linear_reg(pd.Series(dS), pd.Series(x))
            return kappa
        
        kappa = estimate_kappa(spread_series)
        t = np.log(2)/kappa
        return t*time
    
    def mean_cross(self, price_series:np.array, mean:int|float = None, tolerance:int|float = 0.001) -> list[int]:
        """
        Returns a list indicating whenever the price series crosses the mean.
        
        args:
            tolerance: Reduces noise by adding a buffer to ignore volatile movements around the mean.
        returns:
            -1: Cross below the mean
            1: Cross above the mean
            0: No crosses
        """
        if mean is None:
            mean = np.mean(price_series)
        
        cross = 0
        signal = []
        for px in price_series:
            if px > mean+tolerance and cross != 1:
                cross = 1
                signal.append(cross)
            elif px < mean-tolerance and cross != -1:
                cross = -1
                signal.append(cross)
            else:
                signal.append(0)
        
        #Set initial value to 0 as price did not cross the mean
        signal[0] = 0
        return signal

class StockPriceSimulations:
    def __init__(self, initial:float|int=100, days:int=252, volatility:float=0.01):
        self.initial = initial
        self.days = days
        self.volatility = volatility
    
    def init_price_series(self):
        price_series = np.zeros(self.days)
        price_series[0] = self.initial
        return price_series
    
    def vasicek_model(self, speed:float=0.2, mean:float|int=100):
        price_series = self.init_price_series()
        
        if price_series[0] > 0:
            price_series[0] = np.log(price_series[0])
        if mean > 0:
            mean = np.log(mean)
        Z = np.random.normal(0, 1, self.days)
        
        for t in range(1, self.days):
            price_change = speed*(mean-price_series[t-1]) + self.volatility*Z[t-1]
            price_series[t] = price_series[t-1] + price_change
        
        return np.exp(price_series)
    
    def random_walk(self, drift:float=0) -> np.array:
        noise = np.random.normal(0, self.volatility, self.days)
        price_series = self.init_price_series()
        
        for t in range(1, self.days):
            price_series[t] = price_series[t-1] + noise[t] + drift
        
        return price_series
    
    def trend_random_walk(self, trend_strength:float=0.05, direction_up:bool=True) -> np.array:
        direction = 1 if direction_up else -1
        trend = np.array([direction*trend_strength*t for t in range(self.days)])
        price_series = self.init_price_series()
        
        for t in range(1, self.days):
            price_series[t] = price_series[t-1] + trend[t]
        
        return price_series