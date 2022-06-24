from data_processing.image_processing import ProcessDataset
from data_processing.number_processing import NumberProcessing
import os
import numpy as np
import matplotlib.pyplot as plt




class Main(object):
    def __init__(self) -> None:
        self.dataset_processor = ProcessDataset()

    def generate_dataloader(self):
        data_loader = NumberProcessing.create_dataloader(self.X, self.y)

    def dataset_to_numpy(self):
        return self.dataset_processor.get_dataset_as_numpy("classifier\Dataset")

    def load_dataset_as_numpy(self, X_name, y_name):
        X,y = np.load(f"{X_name}.npy"), np.load(f"{y_name}.npy")

        self.X = X
        self.y = y
        return X,y


    def save_dataset_numpy(self, X, y, X_name, y_name):
        print("counting ----------------")
        print(y.count(1))
        print(y.count(2))
        print(y.count(3))
        print(y.count(4))
        np.save(X_name,X)
        np.save(y_name,y)

    


if __name__ == "__main__":
    constr = Main()
    X = list()
    y = list()
    X_name = "Dataset"
    y_name = "labels"
    X,y = constr.dataset_to_numpy()
    constr.save_dataset_numpy(X,y,X_name,y_name)
    #constr.generate_dataloader()