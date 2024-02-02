import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

data_path = '/raid1/ml4ao/torch_format_data/full/'
output_path = '/raid1/ml4ao/torch_data/'

output_data_name = 'tensor_data_olmodes4'
labels_data_name = 'tensor_data_labels4'

filenames= ['comm', 'deltaComm', 'SEEING', 'ATMO_l0', 'ATMO_heigths', 'ATMO_cn2', 'WIND_SPEED', 'srRes']

# save one chunck of prepared data at a time
indata = {}
for idx in range(10):
    for ff in filenames:
        time.sleep(1)
        filename = os.path.join(data_path, ff+'_'+str(idx)+'.pl')
        print(filename)
        if os.path.exists(filename):
            # load and substitute nan with 0
            moreData = torch.nan_to_num(torch.load(filename))
            indata[ff] = moreData
            print(indata[ff].shape)
        else:
            break
    nTNs = indata['SEEING'].shape[0]
    print(nTNs)
    # compute tau0
    r0 = 0.9759 * 0.5 / (indata['SEEING'] * 4.848)
    v = indata['WIND_SPEED']
    wind_speed_average = torch.mean(v, dim=1)
    tau0 = 0.314 * r0 / wind_speed_average * 1e3
    # compute ol_modes
    ol_modes = indata['deltaComm'][:,2:,:100] + indata['comm'][:,:-2,:100]

    labels4 = torch.empty(nTNs, 4)
    labels4[:, 0] = indata['SEEING']
    labels4[:, 1] = indata['ATMO_l0']
    labels4[:, 2] = tau0
    labels4[:, 3] = torch.mean(indata['srRes'][:,20:],dim=1)

    torch.save(ol_modes, os.path.join(output_path, output_data_name) + '_'+str(idx)+'.pt')
    torch.save(labels4, os.path.join(output_path, labels_data_name) + '_'+str(idx)+'.pt')
    
# put together the chuncks of prepared data in two files: one for the inputs, one for the labels

ol_modes = None
labels4 = None
for idx in range(10):    
    ol_modes_i = torch.load(os.path.join(output_path, output_data_name) + '_'+str(idx)+'.pt')
    labels4_i = torch.load(os.path.join(output_path, labels_data_name) + '_'+str(idx)+'.pt')
    if idx==0:
        ol_modes = ol_modes_i
        labels4 = labels4_i
    else:
        ol_modes = torch.cat((ol_modes, ol_modes_i))
        labels4 = torch.cat((labels4, labels4_i))

torch.save(ol_modes, os.path.join(output_path, output_data_name) + '.pt')
torch.save(labels4, os.path.join(output_path, labels_data_name) + '.pt')