import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from trainer.train_cifar_10 import CNN


def main():
    # Loading and normalizing CIFAR10 
    transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar_10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar_10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load the trained model
    param = torch.load('trainer/cifar_10_model.pth')
    model = CNN()
    model.load_state_dict(state_dict=param)
   
    # Test the network on the test data
    # the accuracy of the whole dataset
    whole_correct = 0
    whole_total = 0
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        whole_total += labels.size(0)
        whole_correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * whole_correct / whole_total))

    # the accuracy of the each classes
    class_correct = list(0.0 for i in range(10)) # class_correct [0.0, 0.0, .. 0.0, 0.0] 
    class_total = list(0.0 for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of {0} : {1} %'.format(classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
