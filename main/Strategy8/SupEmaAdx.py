import pandas as pd
import talib
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import os
import datetime
import yfinance as yf

startdate = datetime.datetime(2024, 6, 7)
enddate = datetime.datetime(2024, 8, 4)
stock = 'SOL'
time = '1d'

def Fetchdata():
    dir = os.getcwd()
    data = pd.read_csv(f"{dir}/data/{stock}-USDT/{stock}-USDT_{time}.csv")
    data.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Date', inplace=True)
    return data

class BBRSIADXEMA(Strategy):
    stlo = 980
    tkpr = 1020
    multiplier=3
    atr_var=10
    dema_var=50
    ema1_var=10
    ema2_var=20

    def init(self):
        self.hl2=(self.data.High+self.data.Low)/2
        self.atr=self.I(talib.ATR,self.data.High,self.data.Low,self.data.Close,timeperiod=self.atr_var)
        self.suplow=self.hl2-(self.multiplier*self.atr)
        self.suphigh=self.hl2+(self.multiplier*self.atr)
        self.dema=self.I(talib.DEMA,self.data.Close,timeperiod=self.dema_var)
        self.ema10=self.I(talib.EMA,self.data.Close,timeperiod=self.ema1_var)
        self.ema20=self.I(talib.EMA,self.data.Close,timeperiod=self.ema2_var)

    def next(self):
        if (self.dema[-3]<self.data.Close[-3] and self.suplow[-4]>self.data.Close[-3] and (self.ema10[-2]<self.ema20[-2] and self.ema10[-1]>self.ema20[-1])):
            self.position.close()
            self.buy(sl=(self.data.Close * self.stlo) / 1000, tp=(self.data.Close * self.tkpr) / 1000)

        elif (self.dema[-3]>self.data.Close[-3] and self.suphigh[-4]<self.data.Close[-3] and (self.ema10[-2]>self.ema20[-2] and self.ema10[-1]<self.ema20[-1])):
            self.position.close()
            self.sell(sl=(self.data.Close * self.tkpr) / 1000, tp=(self.data.Close * self.stlo) / 1000)

def walk_forward(strategy, data_full, warmup_bars, lookback_bars, validation_bars, cash=10000000, commission=0):
    stats_master = []
    trades_master = []
    for i in range(lookback_bars + warmup_bars, len(data_full) - validation_bars, validation_bars):
        training_data = data_full.iloc[i - lookback_bars:i]
        validation_data = data_full.iloc[i:i + validation_bars]
        
        bt_training = Backtest(training_data, strategy, cash=cash, commission=commission)
        stats_training = bt_training.optimize(
            stlo=range(960, 990, 10),
            tkpr=range(1010, 1040, 10),
            dema_var=range(40,60,10),
            multiplier=range(2,4,1),
            maximize='Sharpe Ratio',
        )
        
        opt_stlo = stats_training._strategy.stlo
        opt_tkpr = stats_training._strategy.tkpr
        opt_dema = stats_training._strategy.dema_var
        opt_multiplier = stats_training._strategy.multiplier
        
        bt_validation = Backtest(validation_data, strategy, cash=cash, commission=commission)
        stats_validation = bt_validation.run(
            stlo=opt_stlo, tkpr=opt_tkpr,dema_var=opt_dema,multiplier=opt_multiplier
        )
        
        equity_curve = stats_validation['_equity_curve']['Equity']
        trades = stats_validation['_trades']
        trades['Validation_Set']=i
        a = validation_data['Close']
        b = validation_data['Close'].iloc[0]
        c = a / b
        
        plt.figure(figsize=(14, 14))

        plt.subplot(2, 1, 1)
        plt.plot(equity_curve, label='Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.title('Strategy Equity Curve')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(c, label='Normal Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Normalized Equity')
        plt.title('Normal Equity Curve')
        plt.legend()

        graph_path = os.path.join('graph', f'equity_combined_curve_{stock}_{time}_{i}.png')
        plt.savefig(graph_path)
        plt.close()

        stats_master.append(stats_validation)
        trades_master.append(trades)
    
    var1 = pd.DataFrame(stats_master)
    trades_final = pd.concat(trades_master)
    csv_path = os.path.join('csv', f'file_{stock}_{time}.csv')
    csv_path2 = os.path.join('csv2',f'file_{stock}_{time}.csv')
    trades_final.to_csv(csv_path2)
    var1.to_csv(csv_path)
    
    return stats_master

lookback_bars = 600
validation_bars = 200
warmup_bars = 50

data = Fetchdata()
stats = walk_forward(BBRSIADXEMA, data, warmup_bars, lookback_bars, validation_bars)
for stat in stats:
    print(stat)
