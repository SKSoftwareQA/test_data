import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

# Fetch intraday data for a given ticker and interval
def fetch_intraday_data(ticker, interval='15m'):
    data = yf.download(ticker, period="60d", interval=interval)  # Fetch last 60 days for intraday
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

# Fetch intraday data for dhani and ITC
dhani_data = fetch_intraday_data("dhani.NS", interval='15m')
itc_data = fetch_intraday_data("ITC.NS", interval='15m')

# Ensure the data is in the required format
dhani_data.index = pd.to_datetime(dhani_data.index)
itc_data.index = pd.to_datetime(itc_data.index)

# EMA calculation function using numpy
def EMA(array, period):
    alpha = 2 / (period + 1)
    ema = np.empty(len(array))
    ema[0] = array[0]  # Initialize with the first value
    for i in range(1, len(array)):
        ema[i] = alpha * array[i] + (1 - alpha) * ema[i - 1]
    return ema

# McGinley Dynamic calculation function using numpy
def McGinleyDynamic(array, period):
    md = np.empty(len(array))
    md[0] = array[0]  # Initialize with the first value
    for i in range(1, len(array)):
        md[i] = md[i - 1] + (array[i] - md[i - 1]) / min(2 * period, (array[i] / md[i - 1]) ** 4)
    return md

class EmaCrossMcGinley(Strategy):
    ema_short_period = 51
    ema_long_period = 101
    mcginley_period = 21

    def init(self):
        close = self.data.Close
        self.ema_short = self.I(EMA, close, self.ema_short_period)
        self.ema_long = self.I(EMA, close, self.ema_long_period)
        self.mcginley = self.I(McGinleyDynamic, close, self.mcginley_period)

    def next(self):
        if crossover(self.ema_short, self.ema_long) and self.data.Close[-1] > self.mcginley[-1]:
            self.buy()
        elif crossover(self.ema_long, self.ema_short) or self.data.Close[-1] < self.mcginley[-1]:
            self.sell()

# Create Backtest instance for dhani
bt_dhani = Backtest(dhani_data, EmaCrossMcGinley,
                   cash=10000, commission=.002,
                   exclusive_orders=True)

# Create Backtest instance for ITC
bt_itc = Backtest(itc_data, EmaCrossMcGinley,
                  cash=10000, commission=.002,
                  exclusive_orders=True)

# Run backtests and plot results
output_dhani = bt_dhani.run()
bt_dhani.plot()

output_itc = bt_itc.run()
bt_itc.plot()
