import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
import sys
import os
import yaml
import numpy as np

#torch.use_deterministic_algorithms(True)

import matplotlib as mpl

mpl.rcParams["image.cmap"] = 'jet'

max_datset_elements = -1

import matplotlib.pyplot as plt

default_type = torch.float32

SIZE_DEFAULT = 14
SIZE_LARGE = 16
#plt.rc("font", family="Roboto")  # controls default font
#plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch
import mpl_scatter_density

from matplotlib.colors import LinearSegmentedColormap

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def density_scatter(ts, l_ind, x , y,  ax1 = None, ax2 = None, fig = None, sort = True, bins = 20, **kwargs )   :

    data, x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "linear", bounds_error = False)
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    xmin = 0.95 * min(np.min(x), np.min(y))
    xmax = 1.05 * max(np.max(x), np.max(y))
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([xmin, xmax])

    ax1.set_aspect('equal', adjustable='box')
    ax1.scatter( x, y, c=z, cmap='jet', **kwargs )
    ax1.axline( (xmin,xmin),slope=1,linestyle='--',color='red')

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax1, shrink=0.9)
    cbar.ax.set_ylabel('Density (Simulations)')
    
    ax1.set_aspect('equal')    
    err = x-y
    lv = err
    sigma = np.sqrt(np.mean(lv*lv))
    num_bins = 40
    ax2.hist( lv.ravel(), bins=np.linspace(-3.5*sigma, 3.5*sigma, num=num_bins))
    ax2.set_ylabel('Simulations')
    ax2.set_xlabel('Error [' +  ts.data_loader.units[l_ind] + ']')
        
    return ax1
    
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
    def __init__(self, features_size, hidden_size, output_size, num_layers, dropout_level=0.02, mlp_layers=4):
        super(MultivariateGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU Layer
        self.gru = nn.GRU(features_size, hidden_size, num_layers, batch_first=True, dropout=dropout_level)
        # Fully connected layers
        self.fc = nn.ModuleList()
        if type(mlp_layers) is list:
            self.fc.append(nn.Linear(hidden_size, mlp_layers[0]))
            for ll in range(0, len(mlp_layers)-1):
                self.fc.append(nn.Linear(mlp_layers[ll], mlp_layers[ll+1]))            
            self.out_layer = nn.Linear(mlp_layers[-1], output_size)
            self.mlp_layers = len(mlp_layers)
        else:
            for ll in range(mlp_layers):
                self.fc.append(nn.Linear(hidden_size, hidden_size))                
            self.mlp_layers = mlp_layers
            self.out_layer = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.dp = nn.Dropout(p=dropout_level*2)

    def forward(self, x):
        # Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=default_type)
        # Forward pass through GRU layer
        out, _ = self.gru(x, h0)
        # Taking the output of the last time step
        out = out[:, -1, :]
        # Forward pass through the fully connected layer
        for ll in range(self.mlp_layers):
            out = self.fc[ll](out)
            out = self.leaky_relu(out)
            out = self.dp(out)            
        out = self.out_layer(out)
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=default_type)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=default_type)
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

    def __init__(self, model_type, data_path, labels_path, averaging, spacing, limit, features_avg, output_indices, labels_names, units, test_split=0.2, batch_size=32, plotDists=True):
        self.model_type = model_type
        self.data_path = data_path
        self.labels_path = labels_path
        self.test_split = test_split
        self.batch_size = batch_size
        self.features_avg = features_avg
        self.labels_scales = None
        self.train_loader = None
        self.test_loader = None
        self.output_indices = output_indices

        self.averaging = averaging
        self.spacing = spacing
        self.limit = limit
        
        # commenting as this is not used right now
        # gf = [6, 61, 242, 383, 242, 62, 6]
        # self.afilter = 0.001 * np.array(gf)
        
        self.plotDists = plotDists
        self.labels_names = labels_names
        self.units = units
        self.transient_cut = 340 # discard first 340 timesteps, diring which the loop is converging
        
        self._prepare_data_loaders()

    def _load_data(self):
        # Load data and labels from disk
        data = torch.load(self.data_path).to(default_type)
        dataOrig = torch.transpose( data, 1, 2)        
        print('dataOrig.shape', dataOrig.shape)        
        dataOrig = torch.abs(dataOrig[:max_datset_elements, self.transient_cut:, :])
        if len(self.features_avg)>0:
            dataOrigList = []
            avg_n_modes_p = 0
            for avg_i, avg_n_modes in enumerate(self.features_avg):                
                if avg_i>0 and avg_n_modes-avg_n_modes_p>0:
                    # w averaging
                    w = avg_i+1
                    avgLayerI = torch.nn.AvgPool1d(w, stride=w)
                    dataOrigI = avgLayerI(dataOrig[:,:,avg_n_modes_p:avg_n_modes])
                else:
                    # w=1 w-averaging are just the first avg_n_modes
                    dataOrigI = dataOrig[:,:,:avg_n_modes]
                avg_n_modes_p = avg_n_modes
                dataOrigList.append(dataOrigI)        
            dataOrig = torch.cat(dataOrigList, 2)        
        self.features_size = dataOrig.shape[2]
        labels = torch.load(self.labels_path).to(default_type)
        self.labels = labels[:max_datset_elements,self.output_indices]
        # Averaging in the time dimension, 1 means no averaging
        if self.averaging>=2:
            h_space_points = int(self.averaging/2)
            avgLayer = torch.nn.AvgPool1d(2*h_space_points, stride=2*h_space_points)
            self.data = avgLayer(dataOrig.permute(0,2,1)).permute(0,2,1)
        else:
            self.data = dataOrig
        # Sub-sampling in the time dimension, 1 means no averaging
        self.data = self.data[:,:self.limit:self.spacing,:]
        
        print('data.shape', self.data.shape)
        print('labels.shape', self.labels.shape)                

        s1 = torch.sqrt( torch.mean(torch.square(self.data), 1))
        s2 = torch.mean(s1, dim=1)
        olStd, olMean  = torch.std_mean( s2 )
        # remove the outliers data points 
        if self.model_type=='MLP':
            self.data = s1[ s2 < 2.0*olMean, : ] / olMean
            self.labels = self.labels[s2 < 2.0*olMean, :]         
        else:
            print('self.labels_names', self.labels_names)
#            if self.labels_names[0]=='L0_':
#                self.data = self.data[ (self.labels[:,0] < 8.0) * (s2 < 2.0*olMean), :, :]/ olMean
#                self.labels = self.labels[ (self.labels[:,0] < 8.0) * (s2 < 2.0*olMean), :]        
#            else:

            #olMean = 2*olStd + olMean
            #self.data = self.data[s2 < 2.0*olMean, :, :] # / olMean
            #self.labels = self.labels[s2 < 2.0*olMean, :]

        if self.plotDists:
            for ll in range(self.labels.shape[1]):
                lv = self.labels[:, ll].numpy()
                num_bins = 80
                plt.hist( lv.ravel(), bins=np.linspace(np.min(lv), np.max(lv), num=num_bins))
                plt.ylabel('Simulations')
                plt.xlabel('[' +  self.units[ll] + ']')
                plt.title(self.labels_names[ll] + ' distribution')
                plt.show()
                
        print('labels.shape', self.labels.shape)
        print('data.shape', self.data.shape)
        #        self.labels_scales = torch.mean(self.labels, 0) + torch.std(self.labels, 0) 
        self.labels_scales = torch.max(self.labels, 0).values    
        self.inputs_scales = self.data.pow(2).mean(dim=0).sqrt()                
        self.data = self.data/self.inputs_scales        
        self.labels = self.labels / self.labels_scales        
        print("Mean labels after scaling", torch.mean(self.labels, 0))
        
        return TensorDataset(self.data, self.labels)

    def _prepare_data_loaders(self):
        # Load dataset
        self.dataset = self._load_data()
        # Splitting the dataset into training and testing
        self.total_samples = len(self.dataset)
        self.test_size = int(self.test_split * self.total_samples)
        self.train_size = self.total_samples - self.test_size

        torch.manual_seed(5125)
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
        for kk in self.my_data_map.keys():
            object.__setattr__(self, kk, self.my_data_map[kk])
        self.weight_decay=1e-4
        self.output_size = len(self.my_data_map['output_indices'])
        self.labels_names = self.my_data_map['labels_names']
        self.units = self.my_data_map['units']
        
    def saveModel(self):
        torch.save(self.model.state_dict(), os.path.join(self.outputFolder, self.output_file))

    def loadModel(self):
        self.setupModel()
        self.model.load_state_dict(torch.load(os.path.join(self.outputFolder, self.output_file)))
        # self.model.eval()
        
    def loadData(self, plotDists=True):
        # Retrieve train and test loaders
        self.data_loader = TimeSeriesDataLoader(self.model_type,
                                            os.path.join(self.dataFolder, self.dataFile),
                                           os.path.join(self.dataFolder, self.labelsFile),                                                
                                           self.averaging,
                                           self.spacing,
                                           self.limit,
                                           self.features_avg, 
                                           self.output_indices,
                                           self.labels_names, 
                                           self.units, 
                                           test_split=self.splitFactor, batch_size=self.batch_size, plotDists=plotDists)
        self.train_loader = self.data_loader.get_train_loader()
        self.test_loader = self.data_loader.get_test_loader()
        self.features_size = self.data_loader.features_size

    def setupModel(self):
        
        # Initialize the model, loss function, and optimizer
        if self.model_type=='GRU':
            self.model = MultivariateGRU(self.features_size, self.hidden_size, self.output_size, self.hidden_layers, self.dropout, self.mlp_layers).to(device, dtype=default_type)
        elif self.model_type=='RNN':
            self.model = RNNModel(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device, dtype=default_type)
        elif self.model_type=='LSTM':
            self.model = MultivariateLSTM(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device, dtype=default_type)
        elif self.model_type=='MLP':
            self.model = MLPModel(self.features_size, self.hidden_size, self.output_size, self.hidden_layers).to(device, dtype=default_type)
        
        self.metrics = []
        self.metrics.append(nn.MSELoss())
        self.metrics.append(SymmetricMeanAbsolutePercentageError().to(device, dtype=default_type))
    
        
    def trainModel(self, learning_rate=None, weight_decay=0, num_epochs=None, metric_idx=0):
        if learning_rate:
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
        bestLossTest = 1e9
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        # Training loop
        for epoch in range(self.num_epochs):
            running_loss1 = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(device, dtype=default_type) 
                labels = labels.to(device, dtype=default_type)
                # Forward pass
                outputs = self.model(inputs)
                loss = self.metrics[metric_idx](outputs, labels)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss1 += loss.cpu().item()
            running_loss1 =  running_loss1 / (self.data_loader.train_size / self.batch_size)            
            lossTest = self.testModel(metric_idx)            
            print(f'Epoch [{epoch+1}/{self.num_epochs}]], Loss: {loss.item():.6f}, , Loss Val: {lossTest:.6f}')
            if lossTest < bestLossTest and epoch>self.num_epochs//2:
                bestLossTest = lossTest
                self.saveModel()
            
    def testModel(self, metric_idx):
        running_loss2 = 0.0
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputsT, labelsT in self.test_loader:
                inputsT = inputsT.to(device, dtype=default_type)
                labelsT = labelsT.to(device, dtype=default_type)
                outputsT = self.model(inputsT)
                lossV = self.metrics[metric_idx](outputsT, labelsT)
                running_loss2 += lossV.cpu().item()
        running_loss2 =  running_loss2 / (self.data_loader.test_size / self.batch_size)
        return lossV.item()


def plotResults(ts, labelsIndices=None):

    if labelsIndices is None:
        labelsIndices = range(len(ts.labels_names))
    
    for l_ind in labelsIndices:

        print("Results for: ", ts.labels_names[l_ind])
              
        ts.model.eval()
        test_errors = []
        test_labels = []
        test_predictions = []

        running_loss1 = 0.0
        with torch.no_grad():  # No need to track gradients during evaluation
            for i, (inputsT, labelsT) in enumerate(ts.test_loader):
                inputsT = inputsT.to(device, dtype=default_type)
                labelsT = labelsT.to(device, dtype=default_type)[:,l_ind]
                outputsT = ts.model(inputsT)[:,l_ind]
                errorsT = outputsT - labelsT
                test_errors.extend(errorsT.cpu().numpy())
                test_labels.extend(labelsT.view(-1).tolist())
                test_predictions.extend(outputsT.view(-1).tolist())
                lossV = ts.metrics[1](outputsT, labelsT)
                running_loss1 += lossV.cpu().item()

        train_errors = []
        train_labels = []
        train_predictions = []
        with torch.no_grad():  # No need to track gradients during evaluation
            for i, (inputs, labels) in enumerate(ts.train_loader):
                inputs = inputs.to(device, dtype=default_type)
                labels = labels.to(device, dtype=default_type)[:,l_ind]
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

        test_labels *= ts.data_loader.labels_scales[l_ind].item()
        train_labels *= ts.data_loader.labels_scales[l_ind].item()
        test_predictions *= ts.data_loader.labels_scales[l_ind].item()
        train_predictions *= ts.data_loader.labels_scales[l_ind].item()

        train_errors = train_labels - train_predictions        
        test_errors = test_labels - test_predictions

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[24,10])

        fig.suptitle(ts.labels_names[l_ind], fontsize=24)

        ax12 = fig.add_axes([0.294, 0.225, 0.11, 0.2])
        ax22 = fig.add_axes([0.718, 0.225, 0.11, 0.2])

        ax = density_scatter(ts, l_ind, train_labels, train_predictions, ax1, ax12, fig, bins=[15,15])
        ax.set_title('Training Set')
        ax.set_xlabel('Truth ' + '[' +  ts.data_loader.units[l_ind] + ']')
        ax.set_ylabel('Estimate ' + '[' +  ts.data_loader.units[l_ind] + ']')
        ax = density_scatter(ts, l_ind, test_labels, test_predictions, ax2, ax22, fig, bins=[15,15])
        ax.set_title('Test Set')
        ax.set_xlabel('Truth ' + '[' +  ts.data_loader.units[l_ind] + ']')
        ax.set_ylabel('Estimate ' + '[' +  ts.data_loader.units[l_ind] + ']')
        plt.show()

        test_abs_errors = np.abs(test_errors)
        train_abs_errors = np.abs(train_errors)
        print("Mean/Median Absolute error TEST:", np.mean(test_abs_errors), np.median(test_abs_errors))
        print("Mean/Median Absolute error TRAIN:", np.mean(train_abs_errors), np.median(train_abs_errors))
        print("R^2", 1-np.mean(np.square(test_abs_errors))/np.var(test_labels))
        test_labels[test_labels == 0.0] = 1
        train_labels[train_labels == 0.0] = 1
        print("Mean/Median Relative error TEST:", np.mean(np.abs(test_errors)/np.abs(test_labels)), np.median(np.abs(test_errors)/np.abs(test_labels)))
        print("Mean/Median Relative error TRAIN:", np.mean(np.abs(train_errors)/np.abs(train_labels)), np.median(np.abs(train_errors)/np.abs(train_labels)))

    if __name__ == '__main__':
        # it is considered as the model base name
        param_1 = sys.argv[1]
        ts = trainingService(param_1)
        ts.loadData()
        ts.setupModel()
        ts.trainModel()
        ts.testModel()
