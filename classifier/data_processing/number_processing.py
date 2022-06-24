import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split




class NumberProcessing(object):

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def create_dataloader(arr):
        # arr.shape is (n,28,28,3)
        train_data, test_data,train_loabels,test_labels = train_test_split()
        

