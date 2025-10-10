import pandas as pd
import torch
from vmdpy import VMD
from fastprogress import master_bar, progress_bar
import numpy as np
from models.LSTM_Torch import SimpleLSTM
from models.LSTM_TF import LSTMRegressor
from torch.autograd import Variable
from Features import derived_features, sliding_windows 

class Regressor():

    #####  Parameters  ######################
    batch_size=32
    seq_length = 28
    num_epochs = 2000
    learning_rate = 1e-3
    input_size = 1
    hidden_size = 200
    num_layers = 2
    num_classes = 1
    device = 'cpu'

    def __init__(self):
                #####Init the Model #######################
        # lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
        # self.lstm =  LSTMRegressor(seq_len, input_size, output_size=self.num_classes)
        self.lstm = SimpleLSTM(self.input_size, self.hidden_size, self.num_layers)
        self.lstm.to(self.device)
        ##### Set Criterion Optimzer and scheduler ####################
        self.criterion = torch.nn.MSELoss().to(self.device)    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)
        #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    def tensor_inputs(self, x, y):
        train_size = int(len(y) * 0.67)
        test_size = len(y) - train_size

        trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

        testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
        testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

        return trainX, trainY, testX, testY

    # Train the model
    def train_lstm_torch(self, num_epochs, trainX, trainY, testX, testY):
        for epoch in progress_bar(range(num_epochs)):
            self.lstm.train()
            outputs = self.lstm(trainX.to(self.device))
            self.optimizer.zero_grad()

            # obtain the loss function
            loss = self.criterion(outputs, trainY.to(self.device))

            loss.backward()


            self.optimizer.step()

            #Evaluate on test
            self.lstm.eval()
            valid = self.lstm(testX.to(self.device))
            vall_loss = self.criterion(valid, testY.to(self.device))
            self.scheduler.step(vall_loss)

            if epoch % 50 == 0:
                print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))
        
        return self.lstm
    def train_lstm_tf(self, num_epochs=2000, batch_size=32, trainX, trainY, testX, testY):
        history = self.lstm.fit(
            trainX, trainY,
            # validation_data=(testX, testX),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=2
        )

        # Evaluation
        results = model.evaluate(testX, testY, verbose=0)
        print(f"Test Results - MAE: {results[1]}, MSE: {results[2]}, MAPE: {results[3]}")
    def main(self, data):
            prices_scaled = derived_features(prices=data)
            # x, y = sliding_windows(mode_series_scaled, seq_length=28)
            x, y = sliding_windows(prices_scaled, self.seq_length)

            print(f"[INFO] sliding_windows() returned:")
            print(f"       x.shape = {x.shape}  # (num_samples, seq_length, features)")
            print(f"       y.shape = {y.shape}  # (num_samples, 1 or seq_length depending on target prep)")

            trainX, trainY, testX, testY = self.tensor_inputs(x, y)
            
            print(f"[INFO] After tensor_inputs():")
            print(f"       trainX.shape = {trainX.shape}")
            print(f"       trainY.shape = {trainY.shape}")
            print(f"       testX.shape  = {testX.shape}")
            print(f"       testY.shape  = {testY.shape}")

            print("\n[INFO] Model Architecture:\n")
            print(self.lstm)   # This will print a full summary of layers

            # Compare TF vs Torch for future prefrence
            self.lstm = self.train_lstm_torch(self.num_epochs, trainX, trainY, testX, testY)
            # self.lstm = self.train_lstm_tf(self.num_epochs, self.batch_size, trainX, trainY, testX, testY)
            
if __name__ == "__main__":

    data_file = '../data/BTC_USDT-1d.feather'
    df_ohclv = pd.read_feather(data_file)

    app = Regressor()
    app.main(df_ohclv['close'])
    
