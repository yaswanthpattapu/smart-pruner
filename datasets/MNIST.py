#train and test data loder return for MNIST dataset
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MNIST:
    def __init__(self, batch_size=128, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def get_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #downloading train and test sets
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        #setting train data loader and test data loader
        trainloader = torch.utils.data.DataLoader(testset, self.batch_size, shuffle=self.shuffle, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, self.batch_size, shuffle=self.shuffle, num_workers=2)
        
        return trainloader, testloader
        