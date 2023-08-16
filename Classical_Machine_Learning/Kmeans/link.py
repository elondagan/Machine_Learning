from abc import ABC, abstractmethod


class Link(ABC):
    @abstractmethod
    def compute(self, cluster, other, distances_dict):
        """
        returning distance between two clusters
        """
        pass


class SingleLink(Link):
    def compute(self, cluster, other, distances_dict):
        """
        returning distance between two clusters using single link method
        """
        min_distance = -1
        for c_sample in cluster.samples:
            for o_sample in other.samples:
                i = c_sample.s_id
                j = o_sample.s_id
                if min_distance == -1:
                    min_distance = distances_dict[(i, j)]
                if distances_dict[(i, j)] < min_distance:
                    min_distance = distances_dict[(i, j)]
        return min_distance


class CompleteLink(Link):
    def compute(self, cluster, other, distances_dict):
        """
        returning distance between two clusters using complete link method
        """
        max_distance = 0
        for c_sample in cluster.samples:
            for o_sample in other.samples:
                i = c_sample.s_id
                j = o_sample.s_id
                if distances_dict[(i, j)] > max_distance:
                    max_distance = distances_dict[(i, j)]
        return max_distance
