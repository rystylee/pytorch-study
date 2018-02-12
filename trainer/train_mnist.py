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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5) # Conv2d (1, 10, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5) # Conv2d (10, 20, kernel_size=(5, 5), stride=(1, 1))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=320, out_features=50) # Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10) # Linear(in_features=50, out_features=10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(input=self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(input=self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x


def main():
    # Loading and normalizing MNIST
    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root='../data/mnist', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers= 1)

    test_set = torchvision.datasets.MNIST(root='../data/mnist', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True, num_workers= 1)

    model = CNN()

    # Define a optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.5)

    # Train the network
    num_epochs = 2
    for epoch in range(num_epochs): # loop over the dataset multiple times
        model.train()
        for i, data in enumerate(train_loader, 0): # this is same as enumerate(trainloader)
            # get the input
            inputs, targets = data

            # wrap them  in Variable
            inputs, targets = Variable(inputs), Variable(targets)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = F.nll_loss(outputs, targets)
            loss.backward() # Calculate differential coefficient
            optimizer.step() # Update parameters

            # print statistics
            if i % 100 == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}'.format(epoch + 1, i * len(inputs), len(train_loader.dataset), 100.0 * i / len(train_loader), loss.data[0]))

    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')


if __name__ == '__main__':
    main()
