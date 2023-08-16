import pandas
import sample


class Data:

    def __init__(self, path):
        """
        create a dictionary from csv file
        :param path: path to the data base
        """
        df = pandas.read_csv(path)
        self.data = df.to_dict(orient="list")

    def create_samples(self):
        """
        create list of object with Sample type
        :return: list of Samples
        """
        sample_list = []
        temp_list = []
        for index in range(len(self.data['samples'])):
            for key in self.data.keys():
                temp_list.append(self.data[key][index])
            sample_list.append(sample.Sample(temp_list[0], temp_list[2:], temp_list[1]))
            temp_list.clear()
        return sample_list
