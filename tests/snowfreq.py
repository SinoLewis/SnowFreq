import argparse
import requests

parser = argparse.ArgumentParser(description="Snowfreq Quant Bot.")
parser.add_argument("--live", action="store_true", help="Set it to for backtest + live mode or leave blank for backtest only mode.")
args = parser.parse_args()

if args.live:
    print("live is True. Running in live + backtest mode.")

    import time 
        
    while True:
        print("Loop checker!!!") 
        time.sleep(10) 
else:
    print("live is False. Running in backtest mode.")

url = 'http://127.0.0.1:8080/api/v1/'
cmd = 'ping'

r = requests.get(url=f'{url}{cmd}')
print(r)
# r = requests.post(url=API_ENDPOINT, data=data)

## MODULE INJECTION

import pickle
from pprint import pprint
import pandas as pd

# import pickle

# data_file = "/freqtrade/user_data/tests/data/assets_prices.pkl"
# with open(data_file, 'wb') as f:
#     data = pickle.dump(data, f)

# print(f"BT DATA Backup to: {data_file}")
data_dir = "/freqtrade/user_data/tests/data"
# data_dir = "/home/eren/SnowFreq-master/quant-bot/user_data/tests/data"

with open(f'{data_dir}/bt_data.pkl', 'rb') as f:
    data = pickle.load(f)

# pprint(data)

# A. STATIC DESIGN (Backtest)

# 1. Create Test file & data folder from freq data
# 2. docker compose CMD to mngt container w/ vols of above
# 3. Confirm test file run w/ Parsed args: pairs, time(frame/range)

# Multi-Asset Data for Portfolio Optimizers
# TODO: Convert asset df files = dict{asset: df}
close_prices = pd.DataFrame({
    asset: df["close"]
    for asset, df in data.items()
})

pprint(close_prices)

# B. DYNAMIC DESIGN (Live)
# Inside Freqtradebot instance; we can query obj self.dataprovider.get_pair_dataframe to get Asset data. Though Exchange API key required 
# thus, we can use freqtradebot exchange obj to get live data

# UPDATE: use freq API to get RPC REQ: freqtrade-client cmds & REST API endpoint & Web Socket 
# 1. Consume freq API such that we get an updated assets df periodically 4 BT + Live modes
# - Configuration with docker + security key + CORS
# - RPC REQ might be useful
# /whitelist 	GET 	Show the current whitelist.
# /available_pairs 	GET 	List available backtest data. Alpha
# /pair_candles     GET/POST    Returns dataframe for a pair / timeframe combination while the bot is running, Alpha
# /pair_history  	GET/POST 	Returns an analyzed dataframe for a given timerange, analyzed by a given strategy. Alpha
# /stats 	GET 	Display a summary of profit / loss reasons as well as average holding times.
# /profit 	GET 	Display a summary of your profit/loss from close trades and some stats about your performance.

# 2. Error Mngt: Fetch req err, Null records value(s), CORS error
# 3. SnowFreq modules use the df data to produce periodic Quant stats

# C. PseudoCode
# 1. Update config.json w/ REST API Confs
# 2. define API handles: bt_data, live_data, asset_lists, profit_stats
# 3. define bt_only & bt_live modes method of Module execution
# 4. modules consist of 4 apps: tech_strats, quant_stats, ai_strats, assets_optimizers 
# 5. bt_only executes once displaying epoch stats 4rm Modules, by use of API handles
# 6. bt_live executes in a loop, periodically mapping stats of bt & live data 4rm Modules, by use of freq ws or loop service
# 7. final md/html analysis report define by an llm with text & chart info
