# Starting the Backend service for our UI.

## 1. Create user dir

> docker-compose run --rm freqtrade create-userdir --userdir user_data

## 2. Create config file

> docker-compose run --rm freqtrade new-config --config user_data/config.json

## 3. Checking Logs

> docker-compose logs -f

## 4. Download data

> docker-compose run --rm freqtrade download-data --pairs ETH/BTC --exchange binance --days 5 -t 1h

## 5. Backtest

> docker-compose run --rm freqtrade backtesting --config user_data/config.json --strategy SampleStrategy --timerange 20190801-20191001 -i 5m

## 6. Plotting

> docker-compose run --rm freqtrade plot-dataframe --strategy AwesomeStrategy -p BTC/ETH --timerange=20180801-20180805

## 7. Analysis

> docker-compose -f docker/docker-compose-jupyter.yml up

