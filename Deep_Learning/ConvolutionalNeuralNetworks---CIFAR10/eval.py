import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from train_model_q1 import CNN

def eval():

    # load CIFAR10 test data
    transform0 = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615)),
    ])
    batch_size = 250

    test_data = datasets.CIFAR10(root='./data/', train=False, transform=transform0, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # load CNN (trained) network
    cnn = CNN()
    cnn.load_state_dict(torch.load('weights.pkl',map_location=lambda storage, loc:storage))
    criterion = nn.NLLLoss()
    # predict the test model

    used_test_size, correct_test_predictions = 0, 0
    # test_loss_per_batch = []
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        # test_loss_per_batch.append(loss.data)
        _, predicted = torch.max(outputs.data, 1)
        used_test_size += labels.size(0)
        correct_test_predictions += (predicted == labels).sum()

    print("CNN model accuracy on test set:")
    print(f"{100 * correct_test_predictions / used_test_size}% correct predictions")


if __name__ == '__main__':
    eval()