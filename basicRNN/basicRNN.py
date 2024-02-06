import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(5125)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModel(nn.Sequential):
    def __init__(self, features_size, hidden_size, num_classes, num_layers):
        super(MLPModel, self).__init__()
        if type(hidden_size) is list:
            self.num_layers = len(hidden_size)
            self.hidden_sizes  = hidden_size
        else:
            self.num_layers = num_layers
            self.hidden_size  = [hidden_size]*self.num_layers
        self.num_classes = num_classes
        self.features_size = features_size
        self.add_module("dense0", nn.Linear(self.features_size, self.hidden_sizes[0]))
        for i in range(self.num_layers-1):
            self.add_module("dense"+str(i+1), nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
#            self.add_module("act"+str(i+1), nn.LeakyReLU(0.1))
#            self.add_module("act"+str(i+1), nn.SiLU())
            self.add_module("act"+str(i+1), nn.LeakyReLU(0.1))
#            self.add_module("dp"+str(i+1), nn.Dropout(0.01))
        self.add_module("denseO", nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]))
#        self.add_module("dpO", nn.Dropout(0.01))
        self.add_module("output", nn.Linear(self.hidden_sizes[-1], self.num_classes))
    

class MultivariateGRU(nn.Module):
    def __init__(self, features_size, hidden_size, output_size, num_layers):
        super(MultivariateGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU Layer
        self.gru = nn.GRU(features_size, hidden_size, num_layers, batch_first=True, dropout=0.03)
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)
        # Taking the output of the last time step
        out = out[:, -1, :]
        # Forward pass through the fully connected layer
        out = self.fc1(out)

        return out

    
class MultivariateLSTM(nn.Module):
    def __init__(self, features_size, hidden_size, output_size, num_layers):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM Layer
        self.lstm = nn.LSTM(features_size, hidden_size, num_layers, batch_first=True, dropout=0.01)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initializing hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Taking the output of the last time step
        out = out[:, -1, :]
        # Forward pass through the fully connected layer
        out = self.fc(out)
        return out


class RNNModel(nn.Module):

    def __init__(self, features_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(features_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_length, features_size)
        out, _ = self.rnn(x)
        # Take the last time step output for prediction
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class TimeSeriesDataLoader:

    def __init__(self, model_type, data_path, labels_path, timesteps, features_size, timeseries, output_indices, test_split=0.2, batch_size=32, plotDists=True):
        self.model_type = model_type
        self.data_path = data_path
        self.labels_path = labels_path
        self.test_split = test_split
        self.batch_size = batch_size
        self.features_size = features_size
        self.scales = None
        self.timeseries = timeseries
        self.train_loader = None
        self.test_loader = None
        self.output_indices = output_indices
        
        gf = [6, 61, 242, 383, 242, 62, 6]
        self.afilter = 0.001 * np.array(gf)
        self.dest_points = timesteps
        self.plotDists = plotDists
        
        self._prepare_data_loaders()

    def _load_data(self):
        # Load data and labels from disk
        data = torch.load(self.data_path).to(torch.float32)
        
        dataOrig = data[:self.timeseries, 20:, :self.features_size]    
        self.spacing = int(dataOrig.shape[1]/self.dest_points)+1
        print('self.spacing', self.spacing)
        self.h_space_points = int(self.spacing/2)

        labels = torch.load(self.labels_path).to(torch.float32)
        self.labels = labels[:self.timeseries,self.output_indices]
        self.data = dataOrig[:, ::self.spacing, :]
#        self.data = dataOrig[:, 100:100+self.dest_points, :]

        s1 = torch.sqrt( torch.mean(torch.square(dataOrig), 1))
        s2 = torch.mean(s1, dim=1)
        olStd, olMean  = torch.std_mean( s2 )
        # remove the outliers data points 
        if self.model_type=='MLP':
            self.data = s1[ s2 < 2.0*olMean, : ] / olMean
            self.labels = self.labels[s2 < 2.0*olMean, :]         
        else:
#            for ii in range(s1.shape[0]):
#                if s2[ii] > 3.0*olMean:
#                    self.data[ii, :, :] = 0.0
#                    self.labels[ii, :] = 0.0
            self.data = self.data[s2 < 2.0*olMean, :, :] / olMean
            self.labels = self.labels[s2 < 2.0*olMean, :] 

                        
        print('data.shape', self.data.shape)
        print('labels.shape', self.labels.shape)                
#        self.scales = torch.mean(self.labels, 0) + torch.std(self.labels, 0) 
        self.scales = torch.median(self.labels, 0).values * 2.0
        print("labels scales", self.scales)
        self.labels = self.labels / self.scales
        
        if self.plotDists:
            for ll in range(self.labels.shape[1]):
                lv = self.labels[:, ll].numpy()
                num_bins = 20 # <-- Change here - Specify total number of bins for histogram
                plt.hist(lv.ravel(), bins=np.linspace(np.min(lv), np.max(lv), num=num_bins)) #<-- Change here.  Note the use of ravel.
                plt.show()
        
        print("labels scales", torch.mean(self.labels, 0))
        
        return TensorDataset(self.data, self.labels)

    def _prepare_data_loaders(self):
        # Load dataset
        self.dataset = self._load_data()
        # Splitting the dataset into training and testing
        self.total_samples = len(self.dataset)
        self.test_size = int(self.test_split * self.total_samples)
        self.train_size = self.total_samples - self.test_size
        
        self.train_dataset, self.test_dataset = random_split(self.dataset, [self.train_size, self.test_size])
        print('train_dataset', self.train_size)
        print('test_dataset', self.test_size)
        # Creating data loaders for training and testing
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader


class trainingService(object):

    def __init__(self, parametersFile):
        if os.path.exists(parametersFile):
            with open(parametersFile) as f:
                my_yaml_dict = yaml.safe_load(f)        
            self.my_data_map = my_yaml_dict
        self.features_size = self.my_data_map['features_size']
        self.hidden_size = self.my_data_map['hidden_size']
        self.hidden_layers = self.my_data_map['hidden_layers']
        self.output_size = len(self.my_data_map['output_indices'])
        self.output_indices = self.my_data_map['output_indices']
        self.num_epochs = self.my_data_map['num_epochs']
        self.learning_rate = self.my_data_map['learning_rate']
        self.dataFolder = self.my_data_map['dataFolder']
        self.outputFolder = self.my_data_map['outputFolder']        
        self.dataFile = self.my_data_map['dataFile']
        self.labelsFile = self.my_data_map['labelsFile']
        self.labels_names = self.my_data_map['labels_names']
        self.splitFactor = self.my_data_map['splitFactor']
        self.batch_size = self.my_data_map['batch_size']
        self.time_points = self.my_data_map['time_points']
        self.time_series = self.my_data_map['time_series']
        self.model_type = self.my_data_map['model_type']
        self.output_file = self.my_data_map['output_file']
        self.weight_decay=1e-4
        
    def saveModel(self):
        torch.save(self.model.state_dict(), self.output_file)

    def loadModel(self):
        self.setupModel()
        self.model.load_state_dict(torch.load(self.output_file))
        self.model.eval()
        
    def loadData(self, plotDists=True):
        # Retrieve train and test loaders
                      
        self.data_loader = TimeSeriesDataLoader(self.model_type,
                                            os.path.join(self.dataFolder, self.dataFile),
                                           os.path.join(self.dataFolder, self.labelsFile),
                                           self.time_points,
                                           self.features_size, 
                                           self.time_series,
                                           self.output_indices,
                                           test_split=self.splitFactor, batch_size=self.batch_size, plotDists=plotDists)
        self.train_loader = self.data_loader.get_train_loader()
        self.test_loader = self.data_loader.get_test_loader()

    def setupModel(self):
        
        # Initialize the model, loss function, and optimizer
        if self.model_type=='GRU':
            self.model = MultivariateGRU(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device)
        elif self.model_type=='RNN':
            self.model = RNNModel(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device)
        elif self.model_type=='LSTM':
            self.model = MultivariateLSTM(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device)
        elif self.model_type=='MLP':
            self.model = MLPModel(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device)
                    
        self.criterion = nn.MSELoss()
    
        
    def trainModel(self, learning_rate=None, weight_decay=0, num_epochs=None):
        if learning_rate:
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs

        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        # Training loop
        for epoch in range(self.num_epochs):
            running_loss1 = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(device) 
                labels = labels.to(device)
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss1 += loss.cpu().item()
            running_loss1 =  running_loss1 / (self.data_loader.train_size / self.batch_size)
            
            if (epoch+1) % 5 == 0:        
                lossTest = self.testModel()            
                print(f'Epoch [{epoch+1}/{self.num_epochs}]], Loss: {loss.item():.6f}, , Loss Val: {lossTest:.6f}')
            
            
    def testModel(self):
        self.model.eval()
        running_loss2 = 0.0
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputsT, labelsT in self.test_loader:
                inputsT = inputsT.to(device)
                labelsT = labelsT.to(device)
                outputsT = self.model(inputsT)
                lossV = self.criterion(outputsT, labelsT)
                running_loss2 += lossV.cpu().item()
        self.model.train()
        running_loss2 =  running_loss2 / (self.data_loader.test_size / self.batch_size)
        return lossV.item()
#        return test_errors, test_labels, test_predictions


def plotResults(ts, labelsIndices):
    for l_ind in labelsIndices:

        print("Results for: ", ts.labels_names[l_ind])
              
        ts.model.eval()
        test_errors = []
        test_labels = []
        test_predictions = []

        running_loss1 = 0.0
        with torch.no_grad():  # No need to track gradients during evaluation
            for i, (inputsT, labelsT) in enumerate(ts.test_loader):
                inputsT = inputsT.to(device)
                labelsT = labelsT.to(device)[:,l_ind]
                outputsT = ts.model(inputsT)[:,l_ind]
                errorsT = outputsT - labelsT
                test_errors.extend(errorsT.cpu().numpy())
                test_labels.extend(labelsT.view(-1).tolist())
                test_predictions.extend(outputsT.view(-1).tolist())
                lossV = ts.criterion(outputsT, labelsT)
                running_loss1 += lossV.cpu().item()

        train_errors = []
        train_labels = []
        train_predictions = []
        with torch.no_grad():  # No need to track gradients during evaluation
            for i, (inputs, labels) in enumerate(ts.train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)[:,l_ind]
                outputs = ts.model(inputs)[:,l_ind]
                # Compute the error for each sample
                errors = outputs - labels
                train_errors.extend(errors.view(-1).tolist())
                train_labels.extend(labels.view(-1).tolist())
                train_predictions.extend(outputs.view(-1).tolist())
    #            if i==0:
    #                pp = inputs[1].cpu().numpy()
    #                plt.plot(pp)
    #                plt.show()

        #plt.plot(test_errors1)
        #plt.plot(train_errors)
        #plt.show()
        test_labels = np.array(test_labels)
        train_labels = np.array(train_labels)
        test_predictions = np.array(test_predictions)
        train_predictions = np.array(train_predictions)

        plt.scatter(test_labels, test_predictions, label=' Test Set')
        plt.scatter(train_labels, train_predictions, label=' Training Set')
        plt.scatter(test_labels, test_labels, label=' Truth')

        test_abs_errors = np.abs(test_errors)
        train_abs_errors = np.abs(train_errors)
        print("Mean/Median Absolute error TEST:", np.mean(test_abs_errors), np.median(test_abs_errors))
        print("Mean/Median Absolute error TRAIN:", np.mean(train_abs_errors), np.median(train_abs_errors))

        print("R^2", 1-np.mean(np.square(test_abs_errors))/np.var(test_labels))

        test_labels[test_labels == 0.0] = 1
        train_labels[train_labels == 0.0] = 1
        print("Mean/Median Relative error TEST:", np.mean(np.abs(test_errors)/np.abs(test_labels)), np.median(np.abs(test_errors)/np.abs(test_labels)))
        print("Mean/Median Relative error TRAIN:", np.mean(np.abs(train_errors)/np.abs(train_labels)), np.median(np.abs(train_errors)/np.abs(train_labels)))
        #plt.ylim(0, 1)
        #plt.xlim(0, 1)
        plt.legend()
        plt.show()

    if __name__ == '__main__':

        # it is considered as the model base name
        param_1 = sys.argv[1]
        ts = trainingService(param_1)
        ts.loadData()
        ts.setupModel()
        ts.trainModel()
        ts.testModel()
