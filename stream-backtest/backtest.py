import streamlit as st
import subprocess, os
from typing import List, Tuple
from supa import supaclient

FREQ_BIN = '/home/eren/miniconda3/envs/freqtrade/bin/freqtrade'
USER_DIR = '/media/eren/PARADIS/SnowCode/Python/MY-REPOS/SnowFreq/stream-backtest/snow_data' 

strats = os.listdir(f'{USER_DIR}/strategies')
configs = os.listdir(f'{USER_DIR}/configs')

strat_files = st.multiselect('Choose strategy files', strats)
strat_files_no_extension = [os.path.splitext(file_name)[0] for file_name in strat_files]
st.write("Strategy files selected: ", strat_files)

config_file = st.selectbox('Choose config file', configs)
st.write("Config file selected: ", config_file)

def get_strats():
    # TODO: Get class name of strategy
    pass

def backtest(config: str, strategy: List[str], freqai: str, timerange: Tuple[int, int]):
    strategies = " ".join(strategy)
    timeranges = f"{timerange[0]}-{timerange[1]}"
    command = f"{FREQ_BIN} backtesting -c {USER_DIR}/configs/{config} --strategy {strategies} --freqaimodel {freqai} --timerange {timeranges} --userdir {USER_DIR}"
    st.success(command, icon="✅")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        st.success(f"Command Output of backtest", icon="✅" )
        st.text(result.stdout)
    else:
        st.error(f"Command Errors of backtest", icon="🚨")
        st.text(result.stderr)

@st.cache_data
def db_get_backtest(col, id):
    col = '*' if name is None else name
    if id is None:
        response = supaclient.table('backtests').select(col).execute()
    else:
        response = supaclient.table('backtests').select(col).eq('backtests.id', id).execute()
    st.success(response, icon="✅")

@st.cache_data
def db_set_backtest(id, data):
    data, count = supaclient.table('backtests').insert({"id": id, "name": data.trades}).execute()


if st.button('Backtest 📉'):
    backtest(config_file, strat_files_no_extension, 'XGBoostRegressor', (20220501,20220701))

# def genre():
#     st.text(st.session_state.genre)

# st.radio( "What\'s your favorite movie genre", ('Comedy', 'Drama', 'Documentary'), key='genre', on_change=genre)

# if st.button('Supabase 🔎'):
#     supabase()