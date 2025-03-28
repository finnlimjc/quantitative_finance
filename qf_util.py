import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

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

class MovingAverage:
    def simple_ma(self, price_series:pd.Series, window:int=20) -> pd.Series:
        sma = price_series.rolling(window).mean()
        sma.name = f'SMA{window}'
        return sma
    
    def mean_cross_trade(self, price_series:pd.Series, mean:pd.Series, tolerance:float=0.0001) -> pd.Series:
        """
        Mainly to indicate long and short whenever price goes above or below the mean.
        
        args:
            price_series: Price data of an asset.
            mean: Either a constant or a moving average that is the same size as the price_series.
            tolerance: A constant to reduce noise.
            
        return:
            A panda series containing the following values:
                -1: Cross below the mean
                1: Cross above the mean
                0: No crosses
        """
        
        cross = 0
        signal = []
        for px, ma in zip(price_series, mean):
            if px > ma+tolerance and cross != 1:
                cross = 1
                signal.append(cross)
            elif px < ma-tolerance and cross != -1:
                cross = -1
                signal.append(cross)
            else:
                signal.append(0)
        
        #Set initial value to 0 as price did not cross the mean
        signal[0] = 0
        return signal

class TradeBacktest:
    def __init__(self, price_series:pd.Series, cash:int=10000, commission:float=0):
        #Create Dataframe
        price_series.name = 'price'
        self.df = price_series.to_frame()
        self.df['equity'] = None
        self.df.iloc[0, 1] = cash
        
        self.cash = cash
        self.commission = commission
        self.position_details = None
        self.equity_curve = []
        self.win_loss = [] # (Profit, W/L)
        self.bankrupt = False
    
    def record_trade(self, profit:float=0, close:bool=False):
        """
        Updates record with a tuple containing (profit, win/loss).
        Win: 1, Lost: -1, Neutral: 0
        
        args:
            profit: Calculated from get_profit function.
            close: If False, an entry will be (None, None) as the trade is not closed.
        """
        if close:
            win_loss = 1 if profit > 0 else -1
            record_entry = (profit, win_loss)
            self.win_loss.append(record_entry)
        else:
            record_entry = (None, 0)
            self.win_loss.append(record_entry)
    
    def open_position(self, price:float, long_short:int, lot_size:int=10):
        """
        Initialize a position and pays for commission.
        
        args:
            price: How much it costs to purchase one share.
            long_short: 1 is long, -1 is short.
            lot_size: How many shares you wish to purchase.
            
        return:
            Update the cash position and equity curve.
            Update the trade position:
                pos_size: price*lot_size
                lot_size: How many shares you wish to purchase.
                long_short: 1 is long, -1 is short.
        """
        #Check Eligiblity
        pos_size = price*lot_size
        if self.cash < pos_size + self.commission:
            print(f"Not enough capital to purchase shares. \nCurrent Cash: {self.cash} \nPosition Size: {pos_size}")
            self.bankrupt = True
            return
        
        #Update equity curve
        self.cash -= self.commission
        self.equity_curve.append(self.cash)
        self.record_trade() #Update as None because trade has not closed
        
        #Update cash
        self.cash -= pos_size
        self.position_details = (pos_size, lot_size, long_short)
    
    def get_profit(self, price:float) -> float:
        pos_size, lot_size, long_short = self.position_details
        close_size = price*lot_size
        
        if long_short == 1:
            profit = close_size - pos_size
        elif long_short == -1:
            profit = pos_size - close_size
        
        return profit
    
    def close_position(self, price:float) -> float:
        """
        Closes a previously opened position and pays for commission.
        
        args:
            price: How much it costs to purchase one share.
            
        return:
            Updates the cash position, equity curve and record.
            Closes the trade position.
            profit: Profit/Loss resulting from the position, after commission.
        """
        
        profit = self.get_profit(price) - self.commission
        pos_size = self.position_details[0]
        self.cash = self.cash + profit + pos_size
        
        self.equity_curve.append(self.cash)
        self.record_trade(profit, close=True)
        self.position_details = None
    
    def mark_to_market(self, price:float):
        """
        Update the equity curve and record.
        """
        self.record_trade()
        if self.position_details is None:
            self.equity_curve.append(self.cash)
            return
        
        pos_size = self.position_details[0] 
        profit = self.get_profit(price)
        
        current_val = self.cash + profit + pos_size
        self.equity_curve.append(current_val)
    
    def ma_price_cross_strategy(self, entry_signals:pd.Series, lot_size:int=10):
        """
        Execute moving average vs price cross strategy.
        
        args:
            entry_signals: A panda series of 1 and -1, where 1 is long and -1 is short.
            lot_size: How many shares to purchase or short.
            
        return:
            Updates equity curve.
        """
        for signal, price in zip(entry_signals, self.df['price']):
            if signal == 0:
                self.mark_to_market(price)
                continue
            
            if self.position_details is None:
                self.open_position(price, signal, lot_size)
            else:
                self.close_position(price)
                self.equity_curve.pop() #Only include update after opening a new position
                self.open_position(price, signal, lot_size)
                self.win_loss.pop() #Only include update for closing an old position
            
            if self.bankrupt:
                self.declare_bankrupt()
                break
        
        self.df['equity'] = self.equity_curve
        self.df[['profit', 'win']] = self.win_loss
    
    def declare_bankrupt(self):
        """
        Update the remaining rows for equity and record.
        """
        remaining = len(self.df) - len(self.equity_curve)
        equity_remaining_entries = [self.cash]*remaining
        self.equity_curve.extend(equity_remaining_entries)
        
        record_remaining_entries = [(None, None)]*remaining
        self.win_loss.extend(record_remaining_entries)

class TradingMetrics:
    def expected_return_volatility(self, log_return:np.array, t:int=0, downside_volatility:bool=False) -> tuple[float,float|None]:
        """
        Calculates the expected return and volatility of the portfolio.
        
        args:
            return_series: An array of returns in percentage term.
            t: Number of recent trading periods. Default to 0, if you wish to calculate lifetime.
            downside_volatility: If True, downside volatility will be calculated.
            
        return:
            A tuple containing:
                expected_return: Average return for the trading window.
                volatility: Standard deviation of returns or standard deviation of downside risks for the trading window.
        """
        if t != 0:
            t = len(log_return) - t
        arr_return = log_return[t:]
        expected_return = np.mean(arr_return)
        
        if downside_volatility:
            mask = arr_return < 0
            filter_arr_return = arr_return[mask]
            if len(filter_arr_return) == 0:
                return (expected_return, None)
            
            semi_variance = np.sum(np.square(filter_arr_return))/len(arr_return)
            volatility = np.sqrt(semi_variance)
        else:
            volatility = np.std(arr_return)
            
        return (expected_return, volatility)
    
    def sharpe_ratio(self, portfolio_return:float, portfolio_volatility:float, risk_free:float=0, t:int=1) -> float|None:
        """
        Calculate the Sharpe Ratio of the portfolio or strategy.
        
        args:
            portfolio_return: Average return of the portfolio in a given timeframe.
            portfolio_volatility: Standard deviation of portfolio return in a given timeframe.
            risk_free: Typically, the 3-month T-Bill Rate divided by 365. If strategy is intrday, put 0.
            t: Number of trading periods that you wish to scale the Sharpe Ratio. For example, 252 for annualized.
        
        return:
            sharpe_ratio: how much excess return you receive for the extra volatility from the riskier asset, scaled to specified timeframe.
        """
        if portfolio_volatility == 0 or portfolio_volatility is None:
            return None
        
        sharpe_ratio = (portfolio_return-risk_free)/portfolio_volatility    
        scaled_sharpe_ratio = np.sqrt(t)*sharpe_ratio
        return scaled_sharpe_ratio
    
    def sortino_ratio(self, portfolio_return:float, downside_volatility:float, risk_free:float=0, t:int=1) -> float|None:
        """
        Calculate the Sortino Ratio of the portfolio or strategy.
        
        args:
            portfolio_return: Average return of the portfolio in a given timeframe.
            downside_volatility: Standard deviation of downside portfolio return in a given timeframe.
            risk_free: Typically, the 3-month T-Bill Rate divided by 365. If strategy is intrday, put 0.
            t: Number of trading periods that you wish to scale the Sortino Ratio. For example, 252 for annualized.
        
        return:
            sortino_ratio: how much excess return you receive for the extra volatility from the riskier asset, scaled to specified timeframe.
        """
        if downside_volatility == 0 or np.isnan(downside_volatility):
            return None
        
        sortino_ratio = (portfolio_return-risk_free)/downside_volatility    
        scaled_sortino_ratio = np.sqrt(t)*sortino_ratio
        return scaled_sortino_ratio
    
    def get_covariance(self, r1:np.array, r2:np.array) -> float:
        """
        Calculate the covariance between the return of Asset 1 and Asset 2.
        
        args:
            r1: Log Return of Asset 1, excluding the NaN values.
            r2: Log Return of Asset 2, excluding the NaN values.
            
        return:
            Covariance between the return of Asset 1 and Asset 2.
        """
        r_arr = np.vstack([r1,r2])
        cov_matrix = np.cov(r_arr, rowvar=True)
        return cov_matrix[0,1]
    
    def get_beta(self, cov:float, benchmark_variance:float) -> float:
        return cov/benchmark_variance
    
    def treynor_ratio(self, portfolio_return:float, beta:float, risk_free:float=0) -> float:
        """
        Calculate the Treynor Ratio.
        
        args:
            portfolio_return: Average log return of the portfolio.
            beta: Beta of the portfolio.
            risk_free: Typically, the 3-month T-Bill Rate divided by 365. If strategy is intrday, put 0.
        
        return:
            Treynor Ratio for the portfolio.
        """
        return (portfolio_return-risk_free)/beta
    
    def information_ratio(self, portfolio_return:float, benchmark_return:float, tracking_error:float) -> float:
        """
        Calculates the information ratio.
        
        args:
            portfolio_return: Average log return of the portfolio.
            benchmark_return: Average log return of the benchmark.
            tracking_error: The standard deviation of the difference between the log return of the portfolio and the benchmark.
        
        return:
            Information ratio for the portfolio
        """
        return (portfolio_return-benchmark_return)/tracking_error
    
    def get_drawdown(self, price_series:pd.Series, t:int=0) -> list[float]:
        """
        Calculates the drawdowns from the peak in a given time period.
        
        args:
            price_series: A panda series of price values.
            t: If 0, calculate for the lifetime. Else, it will calculate for past t-days.
        
        return:
            drawdown: a list of all the drawdowns with a rolling peak. 
        """
        if t != 0:
            t = len(price_series) - t
            
        peak = price_series.iloc[t]
        drawdown = []
        for equity in price_series[t:].values:
            peak = max(peak, equity)
            d = max(0,((peak-equity)/peak))
            drawdown.append(d)
        return drawdown
    
    def yearly_average_return(self, price_series:pd.Series) -> pd.Series:
        log_price = np.log(price_series.values)
        log_returns = log_price[1:] - log_price[:-1]
        log_series =  pd.Series(log_returns, index=price_series.index[1:])
        return log_series

    def sterling_ratio(self, price_series:pd.Series, risk_free:float=0, t:int=3, recent_year:bool=False) -> float:
        """
        Calculate the Sterling Ratio.
        
        args:
            price_series: Index of datetime objects and prices as values.
            risk_free: Typically, the 3-month T-Bill Rate divided by 365. If strategy is intrday, put 0.
            t: Number of years to include in calculation of average annual maximum drawdown.
            recent_year: If True, will treat the current year as the latest year.
            
        return:
            Sterling Ratio: Return achieved per unit of downside risk.
        """
        log_series = self.yearly_average_return(price_series)
        portfolio_returns = log_series.groupby(log_series.index.year).aggregate('mean')
        annual_drawdowns = price_series.groupby(price_series.index.year).aggregate(self.get_drawdown).apply(lambda x: max(x))
        
        if recent_year:
            portfolio_return = np.mean(portfolio_returns.iloc[-t:].values)
            expected_mdd = np.mean(annual_drawdowns.iloc[-t:].values)
        else:
            portfolio_return = np.mean(portfolio_returns.iloc[-t:-1].values)
            expected_mdd = np.mean(annual_drawdowns.iloc[-t:-1].values) #Exclude recent year
            
        return (portfolio_return - risk_free)/expected_mdd
    
    def get_streaks(self, win_series:pd.Series) -> tuple[np.array, np.array]:
        """
        Get an array of win and lose streaks.
        
        args:
            win_series: A panda series where positive number indicates a win and negative number as a lost.
        
        return:
            A tuple of two arrays containing the durations of each new streak.
        """
        target_data = win_series.loc[(win_series>0) | (win_series<0)]
        target_data = np.sign(target_data.values)
        change_pts = np.insert((np.diff(target_data) != 0), 0, True) #Get change points and set first streak
        streak_ids = np.cumsum(change_pts) #Increase by 1 whenever a new streak begins
        streak_lengths = np.bincount(streak_ids)[1:] #Remove initial 0
        streak_type = target_data[np.where(change_pts)[0]] #Filter array to the indexes of the start of new streaks

        streaks = streak_lengths*streak_type
        win_streaks = streaks[streaks > 0].astype(int)
        lose_streaks = (streaks[streaks < 0]* -1).astype(int) #Reverse Sign
        return (win_streaks, lose_streaks)