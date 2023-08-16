import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from hw1_314734724_train_q1 import NN_function, NN


def evaluate_hw1():
    # load networks parameters
    params = torch.load('weights.pkl')

    # load MNIST data set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
    test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    X, Y = NN_function.data_to_torch(test_data)
    results = NN.predict_and_evaluate(X, Y, params)

    print(f"test accuaracy: {results[0]}")
    print(f"test CE loss: {results[1]}")


evaluate_hw1()
