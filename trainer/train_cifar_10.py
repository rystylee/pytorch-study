import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


# Define a Convolution Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # Conv2d (3, 6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120) # Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84) # Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10) # Linear(in_features=84, out_features=10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Loading and normalizing CIFAR10 
    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data/cifar_10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data/cifar_10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = CNN()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    num_epochs = 2
    for epoch in range(num_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # this is same as enumerate(trainloader)
            # get the input
            inputs, labels = data

            # wrap them  in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels) # this is same as criterion.forward(outputs, labels) 
            loss.backward() # Calculate differential coefficient
            optimizer.step() # Update parameters

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[{0}, {1}] loss: {2:.3f}'.format(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Save model
    torch.save(model.state_dict(), 'cifar_10_model.pth')


if __name__ == '__main__':
    main()
