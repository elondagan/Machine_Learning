import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, to_tensor, targetTensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, categories_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(categories_size + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(categories_size + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)  # creatine one long vector
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def get_initial_hidden(self):
        return torch.zeros(1, self.hidden_size)


def train():

    names_per_lang, langs, langs_size, letters_range, letters_size = get_data()

    train_data = []
    for lang in langs:
        for name in names_per_lang[lang]:
            train_data.append([lang, name])
    random.shuffle(train_data)

    """train the model"""
    rnn = RNN(letters_size, 128, letters_size, langs_size)
    criterion = nn.NLLLoss()
    learning_rate = 0.0005

    epochs = 5
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for i, sample in enumerate(train_data):
            lang_input = to_tensor('category', sample[0])
            word_input = to_tensor('word', sample[1])
            target = targetTensor(sample[1]).unsqueeze_(-1)
            hidden = rnn.get_initial_hidden()
            rnn.zero_grad()

            loss = 0
            for j in range(word_input.size(0)):
                output, hidden = rnn.forward(lang_input, word_input[j], hidden)
                loss += criterion(output, target[j])
            loss.backward()

            for p in rnn.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)

            total_loss += loss.item() / word_input.size(0)

            if (i + 1) % 1000 == 0:
                print(f"{np.round(i / len(train_data) * 100, 0)}% completed of Epoc [{epoch}/{epochs}]")

        loss_history.append(np.round(total_loss / len(train_data), 1))

    # plot loss history
    plt.plot(np.arange(epochs), loss_history)
    plt.title("Negative Log Likelihood Loss as a function of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    import pathlib
    path = pathlib.Path().parent.resolve()
    path = str(path).replace('\\', '/')
    torch.save(rnn.state_dict(), path + '/weights.pkl')


if __name__ == '__main__':
    train()
