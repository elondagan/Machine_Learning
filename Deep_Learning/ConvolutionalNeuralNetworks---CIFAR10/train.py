import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layer = nn.Linear(4 * 4 * 60, 10)
        self.dropout = nn.Dropout(p=0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        out = self.layer0(X)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc_layer(out)
        return self.logsoftmax(out)


def train():
    # Image Preprocessing
    transform0 = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615)),
    ])

    transform1 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.247, 0.2434, 0.2615)),
                                     ])

    transform2 = transforms.Compose([transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.247, 0.2434, 0.2615)),
                                     ])

    # hyper parameters
    batch_size = 250
    learning_rate = 0.003
    epochs = 20

    # load CIFAR10 dataset
    train_data0 = datasets.CIFAR10(root='./data/', train=True, transform=transform0, download=True)
    train_data1 = datasets.CIFAR10(root='./data/', train=True, transform=transform1, download=True)
    train_data2 = datasets.CIFAR10(root='./data/', train=True, transform=transform2, download=True)

    train_data = torch.utils.data.ConcatDataset([train_data0, train_data1, train_data2])
    test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform0, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))

    # model evaluation
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=10, verbose=True)
    criterion = nn.NLLLoss()

    """train the model"""
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(epochs):

        used_train_size,  correct_train_predictions = 0, 0
        train_loss_per_batch = []

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            train_loss_per_batch.append(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of train process
            _, predictions = torch.max(outputs, 1)
            used_train_size += len(labels)
            correct_train_predictions += (predictions == labels).sum()

        # train performance progress
        train_loss_history.append(sum(train_loss_per_batch) / len(train_loss_per_batch))
        train_accuracy_history.append(100 * correct_train_predictions / used_train_size)

        # test performance progress
        used_test_size, correct_test_predictions = 0, 0
        test_loss_per_batch = []
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            test_loss_per_batch.append(loss.data)
            _, predicted = torch.max(outputs.data, 1)
            used_test_size += labels.size(0)
            correct_test_predictions += (predicted == labels).sum()
        test_accuracy_history.append(100 * correct_test_predictions / used_test_size)
        test_loss_history.append(sum(test_loss_per_batch) / len(test_loss_per_batch))
        # update learning rate
        lr_scheduler.step(1 - round(correct_test_predictions.item() / test_data_size))

        print(f"Epoch {epoch+1}/{epochs} completed ")
        print(f"train accuracy: {train_accuracy_history[epoch]}")
        print(f"test accuracy: {test_accuracy_history[epoch]}")

    # save the model
    import pathlib
    path = pathlib.Path().parent.resolve()
    path = str(path).replace('\\', '/')
    torch.save(cnn.state_dict(), path + '/weights.pkl')

    # plot learning process
    train_loss_history = [i.item() for i in train_loss_history]
    train_accuracy_history = [i.item() for i in train_accuracy_history]
    test_loss_history = [i.item() for i in test_loss_history]
    test_accuracy_history = [i.item() for i in test_accuracy_history]

    plt.plot(np.arange(epochs), train_accuracy_history, label='train')
    plt.plot(np.arange(epochs), test_accuracy_history, label='test')
    plt.title("Model Accuracy as a function of Epochs")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    plt.plot(np.arange(epochs), train_loss_history, label='train')
    plt.plot(np.arange(epochs), test_loss_history, label='test')
    plt.title("Model Loss(NLLLoss) as a function of Epochs")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
