#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:10:06 2024

@author: vikaskumar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:42:53 2024

@author: vikaskumar
"""

import schedule
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import warnings
import time
import pytz
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None



warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None




BASE_URL = "https://api.binance.com/api/v3"

def calculate_sma(data, window):
    return data['close'].rolling(window=window).mean()
def convert_timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp / 1000.0)


def fetch_klines(symbol, interval, start_time, end_time):
    url = f"{BASE_URL}/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    
    return data

def downloader():
    # all_data= pd.read_csv('15m_universe_2019.csv')
    symbol = "BTCUSDT"
    intervals = ["30m"]
    start_date = (datetime.now() - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.now()

    # Adjust the end_date to the last second of the day
    end_date = end_date.replace(hour=23, minute=59, second=59)

    delta = timedelta(hours=5, minutes=30)

    base_url = f"https://api.binance.com/api/v1/klines"
    all_data= pd.DataFrame()
    
    current_date = start_date
    while current_date <= end_date:
        for interval in intervals:
            url = f"{base_url}?symbol={symbol}&interval={interval}"
            start_time = int(current_date.timestamp()) * 1000
            end_time = int((current_date + timedelta(days=1)).timestamp()) * 1000
            klines_data = fetch_klines(symbol, interval, start_time, end_time)
            columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
            df = pd.DataFrame(klines_data, columns=columns)
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df['timestamp'] = df['timestamp'].apply(convert_timestamp_to_datetime)
            df['granularity']= interval
            all_data = pd.concat([all_data, df]).sort_values(by='timestamp')
            
        current_date += timedelta(days=1)
    return all_data
all_data= downloader()
all_data= all_data[0:len(all_data)-1]
all_data['current_timestamp'] = pd.to_datetime(datetime.now())
# all_data= all_data[0:len(all_data)-1]
all_data['period']= np.arange(1, len(all_data)+1)

def crossover_tp():
    # all_data= pd.read_csv('btc1h.csv')
    # all_data.rename(columns= {'datetime': 'timestamp'}, inplace= True)

    # all_data= pd.read_csv('universe_30m_2019.csv')
    # all_data= pd.read_csv('15m_universe_2019.csv')
    all_data= downloader()
    all_data['period']= np.arange(1, len(all_data)+1)

    all_data['current_timestamp'] = pd.to_datetime(datetime.now())
    # all_data= all_data[0:len(all_data)-1]
    # all_data['period']= np.arange(1, len(all_data)+1)
    all_data= all_data[0:len(all_data)-1]
    # all_data.to_csv('sample_depl_date.csv', index= False)
    strategy_15m= all_data[['timestamp','open', 'high', 'low', 'close','volume' ]]
            
    strategy_15m['period']= np.arange(1, len(strategy_15m)+1)
    sma_short= 1
    sma_long= 7
   
    strategy_15m['sma_short_period']= sma_short
    strategy_15m['sma_long_period']= sma_long
    strategy_15m['sma_short'] = round(calculate_sma(strategy_15m, sma_short),2)
    strategy_15m['sma_long'] = round(calculate_sma(strategy_15m, sma_long),2)
    
    strategy_15m['close']=strategy_15m['close'].apply(lambda x: round(int(float(x))),2)
    strategy_15m['open']=strategy_15m['open'].apply(lambda x: round(int(float(x))),2)
    strategy_15m['high']=strategy_15m['high'].apply(lambda x: round(int(float(x))),2)
    strategy_15m['low']=strategy_15m['low'].apply(lambda x: round(int(float(x))),2)
    
    
    strategy_df= strategy_15m[['timestamp', 'open', 'high', 'low', 'close','period', 'sma_short', 'sma_long', 'sma_short_period', 'sma_long_period']][35:]
    strategy_df['sma_long_minus_sma_sort']= strategy_df['sma_long'] - strategy_df['sma_short'] 
    strategy_df['sma_long_minus_sma_short_boolean']= strategy_df['sma_long_minus_sma_sort'].apply(lambda x: 1 if x>0 else -1)
    
    strategy_df['is_crossover?']= strategy_df['sma_long_minus_sma_short_boolean'].shift(1)==strategy_df['sma_long_minus_sma_short_boolean']
    strategy_df['signal']=0
    strategy_df['signal'] = strategy_df.apply(
    lambda row: 2 if (row['is_crossover?'] == False and row['sma_long_minus_sma_short_boolean'] == 1) else 
                1 if (row['is_crossover?'] == False and row['sma_long_minus_sma_short_boolean'] == -1) else row['signal'],
    axis=1)    
    
    strategy_df['callstatus']= False
    
    strategy_df['callstatus'] = strategy_df.apply(
    lambda row: True if (row['is_crossover?'] == False and row['sma_long_minus_sma_short_boolean'] == 1) else 
                True if (row['is_crossover?'] == False and row['sma_long_minus_sma_short_boolean'] == -1) else row['callstatus'],
    axis=1)  
    
    
    short_position= strategy_df[strategy_df['signal']== 2]
    
    short_position['perc_pt_to_capture_tp']=-1* 0.005 * short_position['open']
    
    
    short_position['target_price']= short_position['open'] + short_position['perc_pt_to_capture_tp']
    # short_position['callstatus'] = True
    
    
    
    long_position= strategy_df[strategy_df['signal']== 1]
    # long_position['signal']= 1
    
    long_position['perc_pt_to_capture_tp']= 0.005 * long_position['open']
    
    
    long_position['target_price']= long_position['open'] + long_position['perc_pt_to_capture_tp']
    # long_position['callstatus']= True
    
    
    
    all_position= pd.concat([short_position, long_position]).sort_values(by= 'period')
    
    position_file= pd.merge(all_data, all_position[['period', 'signal', 'target_price', 'perc_pt_to_capture_tp', 'callstatus']], on= 'period', how= 'left').fillna(0)
    position_file['low']=position_file['low'].apply(lambda x: round(int(float(x))),2)
    position_file['close']=position_file['close'].apply(lambda x: round(int(float(x))),2)
    position_file['high']=position_file['high'].apply(lambda x: round(int(float(x))),2)
    position_file['open']=position_file['open'].apply(lambda x: round(int(float(x))),2)
    position_file['target_price']=position_file['target_price'].apply(lambda x: round(int(float(x))),2)
    
    
    
    
    
    
    position_file['target_price'] = position_file['target_price'].replace(0.0, method='ffill')
    position_file['signal'] = position_file['signal'].replace(0.0, method='ffill')
    
    position_file['perc_pt_to_capture_tp'] = position_file['perc_pt_to_capture_tp'].replace(0.0, method='ffill')
   
    position_file['close_minus_target']= position_file['close'] - position_file['target_price']
    
    short_positions= position_file[position_file['signal']==2]
 
    short_positions['if_TP_met?']= short_positions['low'] <= short_positions['target_price']
   
    long_positions= position_file[position_file['signal']==1]
   
    long_positions['if_TP_met?']= long_positions['high'] >= long_positions['target_price']
   
    all_positions= pd.concat([short_positions, long_positions]).sort_values(by= 'period')
    
    return all_positions

c= crossover_tp()

def signal_preparator():
    
    
    calculated_file= crossover_tp()
    calculated_file['callstatus'] = calculated_file['callstatus'].astype(bool)
    calculated_file['block'] = (calculated_file['target_price'] != calculated_file['target_price'].shift()).cumsum()
    
    calculated_file['frequency_of_block'] = calculated_file['block'].map(calculated_file['block'].value_counts())
    
    
    calculated_file['tp_hits']= calculated_file['if_TP_met?']==True  
    calculated_file['signal'] = calculated_file.apply(
    lambda row: -1 if (row['tp_hits'] == True and row['signal'] == 1 and not row['callstatus']) else 
                -2 if (row['tp_hits'] == True and row['signal'] == 2 and not row['callstatus']) else row['signal'],
    axis=1)

    calculated_file.drop_duplicates(subset= ['signal', 'target_price'], keep= 'first', inplace= True)
    
    
    calculated_file['signal_label']= calculated_file['signal'].astype(str).shift(1) + calculated_file['signal'].astype(str)
   
    calculated_file['trade_type']= str()

   
    calculated_file['trade_type']= calculated_file.apply(
        lambda row: 'TP Short hits within the block' if (row['tp_hits'] == True and row['signal_label'] == '2.0-2.0') else 
                    'TP Long hits within the block' if (row['tp_hits'] == True and row['signal_label'] == '1.0-1.0') else row['trade_type'],
        axis=1)
    
    calculated_file['single_hits'] = (calculated_file['callstatus'] == True) &(calculated_file['if_TP_met?']==True) & ((calculated_file['signal'] == 1) | (calculated_file['signal'] == 2))
      
    calculated_file['signal_label']= calculated_file['signal'].astype(str).shift(1) + calculated_file['signal'].astype(str)

    # calculated_file.loc[calculated_file['signal_label'] == '1.02.0', 'signal']= 
    
    # for i in range(len(calculated_file) - 1)
    calculated_file.reset_index(inplace= True)
    def adjust_signals(signal_series):
        for i in range(1, len(signal_series)):
            if signal_series[i-1] == 1.0 and signal_series[i] != -1:
                signal_series[i] = -1
            elif signal_series[i-1] == 2.0 and signal_series[i] != -2:
                signal_series[i] = -2
        return signal_series

    calculated_file['signal'] = adjust_signals(calculated_file['signal'])
    
    
  
    calculated_file['signal_label']= calculated_file['signal'].shift(1).astype(str) + calculated_file['signal'].astype(str)
    calculated_file.loc[calculated_file['signal_label'] == '-1.0-2.0', 'signal'] = 0
    calculated_file['signal_label']= calculated_file['signal'].shift(1).astype(str) + calculated_file['signal'].astype(str)

    calculated_file.loc[calculated_file['signal_label'] == '-1.0-1.0', 'signal'] = 0
    
    calculated_file['signal_label']= calculated_file['signal'].shift(1).astype(str) + calculated_file['signal'].astype(str)
    
    calculated_file.loc[calculated_file['signal_label'] == '-2.0-2.0', 'signal'] = 0
    
    calculated_file['signal_label']= calculated_file['signal'].shift(1).astype(str) + calculated_file['signal'].astype(str)
    
    calculated_file.loc[calculated_file['signal_label'] == '-2.0-1.0', 'signal'] = 0
    calculated_file['signal_label']= calculated_file['signal'].shift(1).astype(str) + calculated_file['signal'].astype(str)
    
  
    return calculated_file
s= signal_preparator() 
   


def run():
    # logger.info("Starting ...")
    signal= signal_preparator()
    # signal = pd.DataFrame({'signal': [-2]})
    last_signal_df= signal.iloc[[-1]]
    
    if os.path.exists("order_status_tp_only.txt"):
        with open("order_status_tp_only.txt", "r") as file:
            order_active = file.read().strip() == "True"
    else:
        order_active = False
    
    print("Is Your Order Active? ",order_active)
    

    for index, row in last_signal_df.iterrows():
        
        signal = row['signal']
        target_price= row['target_price']

        if signal == 1 and not order_active:
            
            print('Signal==1')
            ist = pytz.timezone('Asia/Kolkata')

            # Get the current date and time in IST
            current_time_in_ist = datetime.now(ist)
            print(current_time_in_ist)
          

            # url = "https://untrade.io/api/v1/strategy/trade/"

            # Define the headers
            headers = {
                "Content-Type": "application/json",
                # "UnTrade-Api-Key": "49697153-1h4467567-42cb-9420-2253e6677310"
            }
            
            # Define the payload (JSON data)
            payload = {
                # "algorithm": "324a60b1-a90b-46cd-8dea-khkh90b30113ca0a",
                "action": "buy",  # Replace with your desired action (buy/sell/square_off)
                "target_price": target_price,  # Optional: Replace with your target price
                # "stop_loss": "<YOUR-STOP-LOSS-HERE>"  # Optional: Replace with your stop loss
            }

        # Make a POST request
            response = requests.post(url, headers=headers, json=payload)
            print(response.text)
           
    
            order_active = True
        
            with open("order_status_tp_only.txt", "w") as file:
                file.write(str(order_active))
                file.flush()
      
                

        elif signal == 2 and not order_active:
            
            print('Signal==2')
            ist = pytz.timezone('Asia/Kolkata')

            # Get the current date and time in IST
            current_time_in_ist = datetime.now(ist)
            print(current_time_in_ist)
                
            url = "https://untrade.io/api/v1/strategy/trade/create"
    
                # Define the headers
            headers = {
                "Content-Type": "application/json",
                "UnTrade-Api-Key": "49697153-1a44-42cb-9420-2253e6677310"
            }
            
            # Define the payload (JSON data)
            payload = {
                # "algorithm": "324a60b1-a9jghfjh0b-46cd-8dea-90b30113ca0a",
                "action": "sell",  # Replace with your desired action (buy/sell/square_off)
                "target_price": target_price,  # Optional: Replace with your target price
                # "stop_loss": "<YOUR-STOP-LOSS-HERE>"  # Optional: Replace with your stop loss
            }

        # Make a POST request
            response = requests.post(url, headers=headers, json=payload)
            print(response.text)
            
            order_active = True
        
            with open("order_status_tp_only.txt", "w") as file:
                file.write(str(order_active))
                file.flush()
                
                
         

        elif signal == -1 and order_active:    
            try: 
                
                print('Signal==-1')
                ist = pytz.timezone('Asia/Kolkata')

                # Get the current date and time in IST
                current_time_in_ist = datetime.now(ist)
                print(current_time_in_ist)
                
                url = "https://untrade.io/api/v1/strategy/trade/create"
        
                    # Define the headers
                headers = {
                    "Content-Type": "application/json",
                    "UnTrade-Api-Key": "49697153-1a44-42cb-9420-2253e6677310"
                }
                
                # Define the payload (JSON data)
                payload = {
                    "algorithm": "324a60b1-a90b-46cd-8dea-90b30113ca0a",
                    "action": "square_off",  # Replace with your desired action (buy/sell/square_off)
                    # "target_price": target_price,  # Optional: Replace with your target price
                    # "stop_loss": "<YOUR-STOP-LOSS-HERE>"  # Optional: Replace with your stop loss
                }

            # Make a POST request
                response = requests.post(url, headers=headers, json=payload)
                print(response.text)
                
            except:
              pass
            
            order_active = False
            with open("order_status_tp_only.txt", "w") as file:
                file.write(str(order_active))
                file.flush()
        
        elif signal == -2 and order_active:    
            try: 
                
                print('Signal==-2')
                ist = pytz.timezone('Asia/Kolkata')

                # Get the current date and time in IST
                current_time_in_ist = datetime.now(ist)
                print(current_time_in_ist)
                
                url = "https://untrade.io/api/v1/strategy/trade/create"
        
                    # Define the headers
                headers = {
                    "Content-Type": "application/json",
                    # "UnTrade-Api-Key": "496hjghjfj97153-1a44-42cb-9420-2253e6677310"
                }
                
                # Define the payload (JSON data)
                payload = {
                    # "algorithm": "324a6hhvg0b1-a90b-46cd-8dea-90b30113ca0a",
                    "action": "square_off",  # Replace with your desired action (buy/sell/square_off)
                    # "target_price": target_price,  # Optional: Replace with your target price
                    # "stop_loss": "<YOUR-STOP-LOSS-HERE>"  # Optional: Replace with your stop loss
                }

            # Make a POST request
                response = requests.post(url, headers=headers, json=payload)
                print(response.text)
                
            except:
              pass
            
            order_active = False
            with open("order_status_tp_only.txt", "w") as file:
                file.write(str(order_active))
                file.flush()        
        
        if signal==0:
           
            print('No Signal')
            ist = pytz.timezone('Asia/Kolkata')

            # Get the current date and time in IST
            current_time_in_ist = datetime.now(ist)
            print(current_time_in_ist)
                
                
if __name__ == "__main__":
    
    for minute in [0, 30]:
        
        schedule.every().hour.at(f":{minute:02}").do(run)

    while True:
        
        schedule.run_pending()
   
        time.sleep(60)
    
   













