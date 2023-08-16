import torch




def get_data():
    from io import open
    import glob
    import os
    import unicodedata
    import string

    def findFiles(path): return glob.glob(path)

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    # Turn a Unicode string to plain ASCII
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    return category_lines, all_categories, n_categories, all_letters, n_letters


names_per_lang, langs, langs_size, letters_range, letters_size = get_data()


def to_tensor(input_type, input):

    if input_type == 'category':
        index = langs.index(input)
        tensor = torch.zeros(1, langs_size)
        tensor[0][index] = 1
        return tensor

    if input_type == 'letter':
        index = letters_range.index(input)
        tensor = torch.zeros(1, letters_size)
        tensor[0][index] = 1
        return tensor

    if input_type == 'word':
        tensor = torch.zeros(len(input), 1, letters_size)
        for i in range(len(input)):
            letter = input[i]
            tensor[i][0][letters_range.find(letter)] = 1
        return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [letters_range.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(letters_size - 1)  # EOS
    return torch.LongTensor(letter_indexes)
