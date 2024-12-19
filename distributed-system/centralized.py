import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

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

    
    
def train(net, trainloader, epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001, momentum=0.9)
    
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


def load_model():
    return Net().to(DEVICE)


if __name__ == '__main__':
    trainloader, testloader = load_data()
    net = load_model()
    train(net, trainloader, 3)
    loss, accuracy = test(net, testloader)
    print(f'Loss: {loss:.5f}, Accuracy: {accuracy:.3f}')
            
            
            
    
    
    
        
    