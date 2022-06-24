import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split




class NumberProcessing(object):

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def create_dataloader(X ,y):
        # arr.shape is (n,3,64,64)

        X = torch.tensor(X).float()
        y = torch.tensor(y).long()
        train_data, test_data,train_labels,test_labels = train_test_split(X,y,test_size=.1,shuffle=True)
        train_data = TensorDataset(train_data,train_labels)
        test_data = TensorDataset(test_data,test_labels)

        train_loader = DataLoader(train_data,batch_size=10,drop_last=True)
        test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

        return train_loader, test_loader


