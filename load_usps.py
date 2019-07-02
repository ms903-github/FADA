import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
def load_usps(data_per_class, transform = None, batch_size = 64):
    with h5py.File("usps.h5", "r") as hf:
            train = hf.get("train")
            test = hf.get("test")
            datas_tr = []
            datas_te = []
            for num in range(10):
                data_tr = []
                data_te = []
                i = 0
                while True:
                    y_tr = train.get("target")[i]
                    if y_tr == num:
                        data_tr.append([train.get("data")[i].reshape(16, 16), y_tr])
                    if len(data_tr) == data_per_class:
                        
                        break
                    i +=1
                while True:
                    y_te = test.get("target")[i]
                    if y_te == num:
                        data_te.append([test.get("data")[i].reshape(16, 16), y_te])
                    if len(data_te) == data_per_class:
                        break
                    i +=1
                datas_tr = datas_tr + data_tr
                datas_te = datas_te + data_te
            X_tr = []
            y_tr = []
            X_te = []
            y_te = []
            for data, label in datas_tr:
                X_tr.append(data)
                y_tr.append(label)
            for data, label in datas_te:
                X_te.append(data)
                y_te.append(label)
            
    class DatatoDataset(Dataset):
        def __init__(self, data, label, transform = transform):
            self.datas = data
            self.labels = label
            self.transform = transform
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            sample = [torch.unsqueeze(torch.tensor(self.datas[idx]), dim = 0), self.labels[idx]]
            if self.transform:
                sample = self.transform(sample)

            return sample

    usps_train = DatatoDataset(X_tr, y_tr)
    usps_test = DatatoDataset(X_te, y_te)

    return (usps_train, usps_test)

def sort_mnist(dataset, data_per_class):
    dataset_sort = []
    for num in range(10):
        data = []
        i = 0
        while True:
            label = dataset[i][1]
            if label == num:
                data.append([dataset[i][0], label])
            if len(data) == data_per_class:
                break
            i +=1
        dataset_sort = dataset_sort + data

    return(dataset_sort)
