import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from collections import OrderedDict
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)           # applied twice
        self.fc1 = nn.Linear(16 * 4 * 4, 120)    # (28 -4) / 2 = 12, (12 - 4) / 2 = 4 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)              # flatten the feature
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)



fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition data."""
    # 手动下载数据集到本地
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 将数据集划分为多个分区
    indices = list(range(len(dataset)))
    partition_size = len(indices) // num_partitions
    partition_indices = indices[partition_id * partition_size:(partition_id + 1) * partition_size]
    
    # 划分训练和测试数据
    train_indices, test_indices = train_test_split(partition_indices, test_size=0.2, random_state=42)
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=batch_size)
    
    return trainloader, testloader


    
def train(net, trainloader, epoch,testloader):
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001, momentum=0.9)
    net.train()
    for i in range(epoch):
        print("Epoch {} Start training".format(i + 1))
        j = 0
        for image, label in trainloader:
            j += 1
            optimizer.zero_grad()
            output = net(image.to(DEVICE))
            loss = criterion(output, label.to(DEVICE))
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                print("Loss: {:.5f}".format(loss.item()))
    
    val_loss, val_accuracy = test(net, testloader)
    print("Epoch {} finished, val_loss: {:.5f}, val_accuracy: {:.5f}".format(i + 1, val_loss, val_accuracy))
    result = {'val_loss': val_loss, 'val_accuracy': val_accuracy}
    return result

            
            
def test(net, testloader):
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for image, label in testloader:
            output = net(image.to(DEVICE))
            loss += criterion(output, label.to(DEVICE)).item()
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label.to(DEVICE)).sum().item()
            
    return loss/len(testloader.dataset), correct/total
            











