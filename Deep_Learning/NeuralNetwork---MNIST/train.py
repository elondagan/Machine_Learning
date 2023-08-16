import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle


class NN_function():

    @staticmethod
    def data_to_torch(data):
        sample_size = data[0][0].shape[1] * data[0][0].shape[2]
        data_size = len(data)
        X = torch.zeros(data_size, sample_size)
        Y = torch.zeros(data_size)
        for i, sample in enumerate(data):
            X[i], Y[i] = torch.reshape(sample[0], (1, sample_size)), sample[1]
        return X, Y

    @staticmethod
    def expand_labels(Y, options_size):
        expanded_Y = torch.zeros(Y.shape[0], options_size)
        for i in range(Y.shape[0]):
            targets = torch.zeros(options_size)  # + 0.01 / options_size
            targets[int(Y[i].item())] = 1  # - (9 * 0.01) / options_size
            expanded_Y[i] = targets
        return expanded_Y

    @staticmethod
    def extract_labels(probability_labels):
        size = probability_labels.shape[0]
        Y = torch.ones(size) * (-1)
        for i in range(size):
            Y[i] = torch.argmax(probability_labels[i])
        return Y

    @staticmethod
    def sigmoid(x, derivative=False):
        """
        sigmoid on each cell in given matrix/vector
        """
        if derivative:
            return (torch.exp(-x)) / ((1 + torch.exp(-x)) ** 2)
        return 1 / (1 + torch.exp(-x))

    @staticmethod
    def reLU(x, derivative=False):
        """
        for matrix only. each cell is now equal max(0, cell)
        """
        if derivative:
            return (x > 0).long()
        res = torch.zeros(x.shape[0], x.shape[1])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                res[i][j] = max(0, x[i, j])
        return res

    @staticmethod
    def softmax(x, derivative=False):
        """
        for matrices, each row ia handled separatly
        """
        result = torch.zeros(x.shape[0], x.shape[1])
        row_size = x.shape[0]
        if derivative:
            for i in range(row_size):
                exps = torch.exp(x[i] - x[i].max())
                result[i] = exps / torch.sum(exps, axis=0) * (1 - exps / torch.sum(exps, axis=0))
            return result
        for i in range(row_size):
            numerator = torch.exp(x[i])
            denominator = torch.sum(torch.exp(x[i]))
            result[i] = numerator / denominator
        return result

    @staticmethod
    def tanh(x, derivative=False):
        """
        for matrices
        """
        result = torch.zeros(x.shape[0], x.shape[1])
        row_size = x.shape[0]
        if derivative:
            for i in range(row_size):
                result[i] = 4 / ((torch.exp(x[i]) + torch.exp(-x[i])) ** 2)
            return result
        for i in range(row_size):
            result[i] = (torch.exp(x[i]) - torch.exp(-x[i])) / (torch.exp(x[i]) + torch.exp(-x[i]))
        return result

    @staticmethod
    def normalize(matrix, type):
        if type == 'v':  # normalize each vector separately
            result = torch.zeros(matrix.shape[0], matrix.shape[1])
            row_size = matrix.shape[0]
            for i in range(row_size):
                result[i] = matrix[i] / torch.norm(matrix[i])
            return result
        if type == 'm':  # normalize as matrix
            return matrix / torch.norm(matrix)

    @staticmethod
    def cross_entropy_loss(Y_real, Y_predicted):
        loss = 0
        size = len(Y_real)
        for i in range(size):
            loss += torch.log(Y_predicted[i, torch.argmax(Y_real[i]).item()])
        return -loss / size


class NN:

    def __init__(self, neronos_amount=(28 * 28, 128, 10), epochs=5, learning_rate=0.01, batch_size=100):
        # free parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        # network structure
        self.input_layer_size = neronos_amount[0]
        self.hidden1_size = neronos_amount[1]
        # self.hidden2_size = neronos_amount[2]
        self.output_layer_size = neronos_amount[2]
        # weights initializing
        self.params = {
            'W1': torch.rand(self.input_layer_size, self.hidden1_size),  # 784x128
            'b1': torch.rand(1, self.hidden1_size),  # 1x128
            'W2': torch.rand(self.hidden1_size, self.output_layer_size),  # 128x10
            'b2': torch.rand(1, self.output_layer_size),  # 1x10
        }
        # normalize all weights
        for key, val in self.params.items():
            self.params[key] = NN_function.normalize(self.params[key], 'm')

    def forward_pass(self, X_train):

        self.params['A0'] = X_train
        params = self.params
        # input layer to hidden_1
        params['Z1'] = torch.matmul(params['A0'], params['W1']) + params['b1']
        params['A1'] = NN_function.tanh(params['Z1'])
        # hidden_1 to output_layer
        params['Z3'] = torch.matmul(params['A1'], params['W2']) + params['b2']
        Y_prediction = NN_function.softmax(params['Z3'])

        return Y_prediction  # Nx10

    def backward_pass(self, y_real, y_predicted):
        params = self.params
        changes = {}

        # calculate derivatives
        dl_dz2 = (y_predicted - y_real) / self.batch_size  # 100x10
        dl_da1 = torch.matmul(dl_dz2, params['W2'].T)  # 100x128
        dl_dz1 = dl_da1 * NN_function.tanh(params['A1'], derivative=True)  # 100x128

        changes['W1'] = torch.matmul(params['A0'].T, dl_dz1)
        changes['b1'] = torch.matmul(torch.ones(1, self.batch_size), dl_dz1)
        changes['W2'] = torch.matmul(self.params['A1'].T, dl_dz2)
        changes['b2'] = torch.matmul(torch.ones(1, self.batch_size), dl_dz2)

        # update all weights
        for key, val in changes.items():
            self.params[key] -= self.lr * val  # W_t+1 = W_t - lr*Delta_W_t

    @staticmethod
    def compute_accuracy(Y_real, Y_predicted):
        data_size = len(Y_real)
        Y_hat = NN_function.extract_labels(Y_predicted)
        return sum(Y_hat == Y_real).item() / data_size

    def train_q1(self, X_train, Y_train, X_test, Y_test):
        data_size = len(Y_train)
        expanded_Y = NN_function.expand_labels(Y_train, self.output_layer_size)
        train_acc_per_epoc = []
        test_acc_per_epoc = []
        for epoc in range(self.epochs):

            for i in range(int(data_size / self.batch_size)):  # assuming data_size%batch_size = 0
                curX = X_train[self.batch_size * i: self.batch_size * (i + 1), :]
                curY = expanded_Y[self.batch_size * i: self.batch_size * (i + 1), :]
                output = self.forward_pass(curX)
                self.backward_pass(curY, output)

            train_predictions = self.forward_pass(X_train)
            test_predictions = self.forward_pass(X_test)
            train_acc_per_epoc.append(NN.compute_accuracy(Y_train, train_predictions))
            test_acc_per_epoc.append(NN.compute_accuracy(Y_test, test_predictions))

        return train_acc_per_epoc, test_acc_per_epoc

    def train_q2(self, X, Y, X_test, Y_test):
        data_size = len(Y)
        expanded_Y = NN_function.expand_labels(Y, self.output_layer_size)
        acc_per_epoc_train, acc_per_epoc_test = [], []
        loss_per_epoc_train, loss_per_epoc_test = [], []

        for epoc in range(self.epochs):

            for i in range(int(data_size / self.batch_size)):
                curX = X[self.batch_size * i: self.batch_size * (i + 1), :]
                curY = expanded_Y[self.batch_size * i: self.batch_size * (i + 1), :]
                output = self.forward_pass(curX)
                self.backward_pass(curY, output)

            if (epoc + 1) % 10 == 0:
                train_predictions, test_predictions = self.forward_pass(X), self.forward_pass(X_test)

                # train
                acc_per_epoc_train.append(self.compute_accuracy(Y, train_predictions))
                loss_per_epoc_train.append(
                    NN_function.cross_entropy_loss(NN_function.expand_labels(Y, self.output_layer_size),
                                                   train_predictions))
                # test
                acc_per_epoc_test.append(self.compute_accuracy(Y_test, test_predictions))
                loss_per_epoc_test.append(
                    NN_function.cross_entropy_loss(NN_function.expand_labels(Y_test, self.output_layer_size),
                                                   test_predictions))

        return acc_per_epoc_train, acc_per_epoc_test, loss_per_epoc_train, loss_per_epoc_test

    def predict(self, X, Y):
        data_size = len(Y)
        Y_predicted = NN_function.extract_labels(self.forward_pass(X))
        acc = sum(Y_predicted == Y) / data_size
        print(f" predictions accuracy: {acc}")

    @staticmethod
    def predict_and_evaluate(X, Y, weights):
        params = weights

        # input layer to hidden_1
        params['Z1'] = torch.matmul(X, params['W1']) + params['b1']
        params['A1'] = NN_function.tanh(params['Z1'])
        # hidden_1 to output_layer
        params['Z3'] = torch.matmul(params['A1'], params['W2']) + params['b2']
        Y_prediction = NN_function.softmax(params['Z3'])

        acc = NN.compute_accuracy(Y, Y_prediction)
        loss = NN_function.cross_entropy_loss(NN_function.expand_labels(Y, 10), Y_prediction).item()
        return acc, loss


if __name__ == "__main__":
    batch_size = 100

    # Image Preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

    # MNIST dataset
    train_data = dsets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transform)

    # build neural network
    nn = NN(epochs=10)

    # convert data to torch
    X_train, Y_train = NN_function.data_to_torch(train_data)
    X_test, Y_test = NN_function.data_to_torch(test_data)

    # train the model
    train_acc, test_acc = nn.train_q1(X_train, Y_train, X_test, Y_test)

    # plot train process
    plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label='train')
    plt.plot(np.arange(1, len(train_acc) + 1), test_acc, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('train accuracy')
    plt.title('Train and Test accuracy as a function of Epochs')
    plt.xticks(np.arange(1, 11))
    plt.legend()
    plt.show()

    # # final test of the model
    print("predicting test data...")
    nn.predict(X_test, Y_test)

    # save model parameters
    with open('weights.pkl', 'wb') as handle:
        pass
    torch.save(nn.params, 'weights.pkl')
