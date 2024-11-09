import pickle
from pprint import pprint

test_dir = "/freqtrade/user_data/tests"

with open(f'{test_dir}/bt_strat_content.pkl', 'rb') as f:
    df_loaded = pickle.load(f)

bt_df = df_loaded['results']
pprint(bt_df.columns)

with open(f'{test_dir}/strat_df.pkl', 'rb') as f:
    df_strat = pickle.load(f)
    pprint(df_strat.columns)
