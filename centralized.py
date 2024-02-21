import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
def train(net, trainloader,epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001, momentum=0.9)
    
    for i in range(epoch):
        for image, label in trainloader:
            optimizer.zero_grad()
            output = net(image.to(DEVICE))
            loss = criterion(output, label.to(DEVICE))
            loss.backward()
            optimizer.step()
            
            
def test(net, testloader):
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
            
            
            
def load_data():
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


def load_model():
    return Net().to(DEVICE)


if __name__ == '__main__':
    
    trainloader, testloader = load_data()
    net = load_model()
    train(net, trainloader, 3)
    loss, accuracy = test(net, testloader)
    print(f'Loss: {loss:.5f}, Accuracy: {accuracy:.3f}')
            
            
            
    
    
    
        
    