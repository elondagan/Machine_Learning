class Cluster:
    def __init__(self, c_id, samples):
        """
        :param c_id: cluster id (int)
        :param samples: list of samples with Sample type
        """
        self.c_id = c_id
        self.samples = samples

    def merge(self, other):
        """
        adding cluster other to cluster self and sort self by id
        :param other: cluster
        """
        self.c_id = min(self.c_id, other.c_id)
        self.samples.extend(other.samples)
        self.samples.sort(key=lambda x: x.s_id, reverse=False)

    def compute_avg_dist(self, sample_to_check, distances_dict):
        """
        compute average dist between a cluster to a sample,
        both for inner and outer sample
        :return:  avg dist
        """
        total_sum = 0
        belongs = 0  # checks if this sample belongs to cluster
        i = sample_to_check.s_id
        for sample in self.samples:
            if sample != sample_to_check:
                j = sample.s_id
                total_sum += distances_dict[(i, j)]
            else:
                belongs = 1
        return total_sum / (len(self.samples) - belongs)

    def find_dominant_label(self):
        """
        find the appeared label
        :return: dominant label
        """
        label_list = []
        for sample in self.samples:
            label_list.append(sample.label)
        if len(label_list) == 1:
            return label_list[0]
        label_list.sort()
        counter_1, counter_2, tracker = 1, 0, 0
        pos_1, pos_2 = 0, 0
        dominant_label = "blank"
        for i in range(1, len(label_list)):
            if label_list[i] == label_list[i - 1]:
                counter_1 += 1
                pos_1 = i - 1
            if label_list[i] != label_list[i - 1] or i == len(label_list)-1:
                if counter_2 <= counter_1:
                    if counter_2 < counter_1:
                        pos_2 = i - 1
                        counter_2 = counter_1
                        dominant_label = label_list[pos_2]
                    else:
                        counter_2 = counter_1
                        dominant_label = min(label_list[pos_2], label_list[pos_1])
                counter_1 = 1
        if dominant_label == "blank":
            return label_list[0]
        return dominant_label

    def print_details(self, silhouette):
        """
        print clusters id's, dominant label, silhouette
        :param silhouette: cluster silhouette value
        """
        s_id_list = []
        for sample in self.samples:
            s_id_list.append(sample.s_id)
        print(f"Cluster {self.c_id}: {s_id_list}, dominant label = {self.find_dominant_label()}, "
              f"silhouette = {round(silhouette,3)}")
