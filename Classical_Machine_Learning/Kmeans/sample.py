import math


class Sample:

    def __init__(self, s_id, genes, label):
        """
        :param s_id: sample id (int)
        :param genes: list of genes values
        :param label: type of leukemia (string)
        """
        self.s_id = s_id
        self.genes = genes
        self.label = label

    def compute_euclidean_distance(self, other):
        """
        calculating Euclidean distance between two samples
        :param other: specific sample
        :return: Euclidean distance
        """
        total_sum = 0
        for i in range(len(self.genes)):
            total_sum += (self.genes[i] - other.genes[i])**2
        return math.sqrt(total_sum)



