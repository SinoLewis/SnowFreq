import tensorflow as tf
from models.LSTM_Torch import LSTMRegressor
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn import preprocessing  # pip install sklearn ... if you don't have it!
from collections import deque
import numpy as np
import random
import time

SEQ_LEN = 20  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "BTC_USDT"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def preprocess_df(df):
    df = df.drop("future", axis=1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic. Those nasty NaNs love to creep in.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), np.array(y)  # return X and y...and make X a numpy array!

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0
def build_data(data_dir):
    main_df = pd.DataFrame() # begin empty
    cols=['date', 'low', 'high', 'open', 'close', 'volume']
    ratios = ["BTC_USDT", "ETH_USDT", "BAT_USDT", "BNB_USDT", "DOGE_USDT", "MINA_USDT"]  # the 4 ratios we want to consider
    for ratio in ratios:  # begin iteration
        print(ratio)
        dataset = f'{data_dir}{ratio}-1h.feather'  # get the full path to the file.
        df = pd.read_feather(dataset)  # read in specific file
        df["time"] = df["date"].astype("int64") // 10**9

        # rename volume and close to include the ticker so we can still which close/volume is which:
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

        df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

        if len(main_df)==0:  # if the dataframe is empty
            main_df = df  # then it's just the current df
        else:  # otherwise, join this data to the main one
            main_df = main_df.join(df)

    main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

    main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    main_df.dropna(inplace=True)
    # print(main_df.head())  # how did we do??
    return main_df

def build_model(seq_len, input_size, output_size):
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    model = LSTMRegressor(seq_len, input_size, output_size).model
    model.summary()
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model

def train_model(model, train_x, train_y, validation_x, validation_y):
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}.keras"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard, checkpoint],
    )

    return history
def evaluate_and_save(model, validation_x, validation_y):
    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(NAME))

def main():
    data_dir = '../data/binance/'
    main_df = build_data(data_dir)
    main_df.dropna(inplace=True)

    times = sorted(main_df.index.values)  # get the times
    last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the last 5% of the times

    validation_main_df = main_df[(main_df.index >= last_5pct)]  # make the validation data where the index is in the last 5%
    main_df = main_df[(main_df.index < last_5pct)]  # now the main_df is all the data up to the last 5%
    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

    # print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    # print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    # print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
    input_size = 5    # e.g., OHLCV features
    output_size = 1   # 2 classes

    model = build_model(SEQ_LEN, input_size, output_size)
    train_model(model, train_x, train_y, validation_x, validation_y)
    evaluate_and_save(model, validation_x, validation_y)

if __name__ == "__main__":
    main()