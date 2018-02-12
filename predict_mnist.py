import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn.functional as F

from trainer.train_mnist import CNN


def main():
    # Loading and normalizing MNIST
    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers= 1)

    test_set = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True, num_workers= 1)

    # Load the trained model
    param = torch.load('trainer/mnist_model.pth')
    model = CNN()
    model.load_state_dict(state_dict=param)

    # Test the network on the test data
    model.eval()
    test_loss = 0
    correct = 0
    for data in test_loader:
        images, targets = data
        images, targets = Variable(images, volatile=True), Variable(targets)
        output = model(images)
        test_loss += F.nll_loss(output, targets, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Average loss: {0:.4f}, Accuracy: {1}/{2} ({3:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
