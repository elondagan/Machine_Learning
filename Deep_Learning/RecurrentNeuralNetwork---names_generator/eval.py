import torch
from utils import get_data, to_tensor
from train import RNN


def eval(source_country, first_letter):

    names_per_lang, langs, langs_size, letters_range, letters_size = get_data()

    # load RNN (trained) model
    rnn = RNN(letters_size, 128, letters_size, langs_size)
    rnn.load_state_dict(torch.load('weights.pkl', map_location=lambda storage, loc:storage))

    # generate name
    max_name_length = 20
    with torch.no_grad():  # no backward necessary

        lang_tensor = to_tensor('category', source_country)
        letter_tensor = to_tensor('letter', first_letter)
        t_hidden = torch.zeros(1, 128)

        output_name = first_letter

        for t in range(max_name_length):
            t_output, t_hidden = rnn(lang_tensor, letter_tensor, t_hidden)

            _, next_letter_index = t_output.topk(1)
            next_letter = letters_range[next_letter_index[0][0]]
            if next_letter == letters_range[-1]:  # end of name
                break
            output_name += next_letter
            letter_tensor = to_tensor('letter', next_letter)

        return output_name


if __name__ == '__main__':
    print(eval("Italian", 'M'))

