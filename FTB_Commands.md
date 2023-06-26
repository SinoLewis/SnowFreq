# Running the server

## Build Contaner
> docker-compose up 

## Log in container
> docker exec -it ftb-lewis bash

## Create user dir

> freqtrade create-userdir --userdir user_data

## Create config file

> freqtrade new-config --config user_data/config.json

## New Strategy

> freqtrade new-strategy -S strat_name --template 

Use a template which is either `minimal`, `full` (containing multiple sample indicators) or `advanced`. Default: `full`.

## Use static pairlisting

Edit in config

## Download data
> freqtrade download-data -p PAIRS --days --exchange --datadir -c configuration_file 

## Run Backtest

> freqtrade backtesting --strategy PTLStrategy 

## Finding better Strategies

### Part time larry strats.
1. ReinforcedScalpStrategy.py
      - Use adx indicator alone
      - Has the backtest avg profit % beat the market change %

## Beat the market

1. Change timeframe from 1m to 15m
2. Change timerange 20220101-20220714

## Determine range of avg profit & mkt change

Calculate from report.

# Plotting
docker up -d (develop_plot)

## Run Plotting

> freqtrade plot --strategy ReinforcedScalpStrategy

## Analyse plot


# Hyperopt

> freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy Strategy004 -i 5m -e 2 --config user_data/configs/config.json
Hyperopt will first load your data into memory and will then run populate_indicators() once per Pair to generate all indicators, unless --analyze-per-epoch is specified.

Hyperopt will then spawn into different processes (number of processors, or -j <n>), and run backtesting over and over again, changing the parameters that are part of the --spaces defined.

For every new set of parameters, freqtrade will run first populate_entry_trend() followed by populate_exit_trend(), and then run the regular backtesting process to simulate trades.

After backtesting, the results are passed into the loss function, which will evaluate if this result was better or worse than previous results.

Based on the loss function result, hyperopt will determine the next set of parameters to try in the next round of backtesting.

# Trade 
> docker-compose up
> freqtrade trade --logfile /freqtrade/user_data/logs/freqtrade.log
      --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite
      --config /freqtrade/user_data/config.json
      --strategy SampleStrategy 

## Explore the UI


# Analysis

# FreqAI
1. Live/ Dry run.
> freqtrade trade --strategy FreqaiExampleStrategy --config config_freqai.example.json --freqaimodel LightGBMRegressor

2. Backtesting
> freqtrade backtesting --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel LightGBMRegressor --timerange 20210501-20210701