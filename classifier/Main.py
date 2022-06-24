from data_processing.image_processing import ProcessDataset
from data_processing.number_processing import NumberProcessing
from models import Models
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


class Main(object):
    def __init__(self) -> None:
        self.dataset_processor = ProcessDataset()

    def generate_dataloader(self):
        self.data_loader,self.test_loader = NumberProcessing.create_dataloader(self.X, self.y)

    def dataset_to_numpy(self):
        return self.dataset_processor.get_dataset_as_numpy("classifier\Dataset")

    def load_numpy_dataset(self, X_name, y_name):
        X,y = np.load(f"{X_name}.npy"), np.load(f"{y_name}.npy")

        self.X = X
        self.y = y
        print(X.shape)
        print(y.shape)
        return X,y


    def save_dataset_numpy(self, X, y, X_name, y_name):
        print("counting ----------------")
        print(y.count(1))
        print(y.count(2))
        print(y.count(3))
        print(y.count(4))
        print(X.shape)

        np.save(X_name,X)
        np.save(y_name,y)
    

    def train(self, net, lossfun, optimizer, epochs):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        net.to(device)
        losses = list()
        accuracy = list()

        for epochi in range(epochs):
            i = 0
            batch_acc = list()
            batch_loss = list()
            for X,y in self.data_loader:
                #print(i)
                i+=1
                if i%50==0: print(f"epoch {epochi} and batch {i}")

                X = X.to(device)
                y = y.to(device)
                y = y-1
                yHat = net(X)
                loss = lossfun(yHat,y)
                
                optimizer.zero_grad()
        
                loss.backward()
           
                optimizer.step()
         

                batch_loss.append(loss.item())

                matches = (torch.argmax(yHat,axis=1)==y).float()
                batch_acc.append(torch.mean(matches)*100)
            
            batch_acc = torch.tensor(batch_acc).to('cpu')
            batch_loss = torch.tensor(batch_loss).to('cpu')

            accuracy.append(torch.mean(batch_acc))
            losses.append(torch.mean(batch_loss))
        
        self.accuracy = accuracy
        self.losses = losses
        return net

    def plot_lists(self,*args):

        a = (len(args))
        print("args length")
        print(a)
        fig,ax = plt.subplots(1,a)

        #plt.plot(*args)
        i = 0
        for list in args:
            ax[i].plot(list,'o')
            i+=1

        plt.show()



    
def Initial():
    constr = Main()
    X = list()
    y = list()
    X_name = "Dataset"
    y_name = "labels"
    X,y = constr.dataset_to_numpy()
    constr.save_dataset_numpy(X,y,X_name,y_name)


def Generate():
    constr = Main()
    X = list()
    y = list()
    X_name = "Dataset"
    y_name = "labels"

    constr.load_numpy_dataset(X_name,y_name)
    constr.generate_dataloader()

    net,lossfun,optimizer = Models.model1()

    net = constr.train(net,lossfun,optimizer,100)
    torch.save(net,"network1")

    constr.plot_lists(constr.accuracy,constr.losses)



if __name__ == "__main__":
    Generate()
  
 