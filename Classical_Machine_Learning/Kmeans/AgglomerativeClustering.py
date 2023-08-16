import cluster


class AgglomerativeClustering:

    def __init__(self, link, samples, distances_dict):
        """
        :param link: SingleLink or CompleteLink
        :param samples: list of Samples, containing all samples in data base
        """
        self.link = link
        self.clusters = []
        for sample in samples:
            new_cluster = cluster.Cluster(sample.s_id, [sample])
            self.clusters.append(new_cluster)
        self.distances_dict = distances_dict

    def compute_silhoeutte(self, distance_dict):
        """
        creating dictionary of silhoeutte values
        :param distance_dict: dictionary that holds all distances between all samples
        :return: dictionary of silhoeutte for each sample
        """
        silhoeutte_dict = {}
        for cluster_i in self.clusters:
            if len(cluster_i.samples) == 1:
                silhoeutte_dict[cluster_i.samples[0].s_id] = 0
                continue
            for sample in cluster_i.samples:
                silhoeutte_dict[sample.s_id] = self.compute_sample_sil(sample, cluster_i, distance_dict)
        return silhoeutte_dict

    def compute_sample_sil(self, sample_check, cluster_of, distance_dict):
        """
        compute sample silhoeutte
        :param sample_check: sample to check
        :param cluster_of: cluster of the sample
        :param distance_dict: dictionary that holds all distances between all samples
        :return: s single sample silhoeutte
        """
        out_num = -1
        in_num = cluster_of.compute_avg_dist(sample_check, distance_dict)
        for cluster_i in self.clusters:
            if cluster_i != cluster_of:
                if out_num == -1:
                    out_num = cluster_i.compute_avg_dist(sample_check, distance_dict)
                else:
                    if cluster_i.compute_avg_dist(sample_check, distance_dict) < out_num:
                        out_num = cluster_i.compute_avg_dist(sample_check, distance_dict)
        return (out_num - in_num) / max(out_num, in_num)

    def compute_summery_silhoeutte(self, distance_dict):
        """
        compute summery silhoeutte
        :param distance_dict: dictionary that holds all distances between all samples
        :return: dictionary of silhoeutte for each cluster and totals
        """
        sample_silhoeutte = self.compute_silhoeutte(distance_dict)
        summary_dict = {}
        total_sum = 0  # sums all the cluster_i silhoeuttes
        for cluster_i in self.clusters:
            sample_sum = 0
            for sample in cluster_i.samples:
                sample_sum += sample_silhoeutte[sample.s_id]
            summary_dict[cluster_i.c_id] = sample_sum / len(cluster_i.samples)
            total_sum += sample_sum
        counter = 0
        for cluster_i in self.clusters:
            for _ in cluster_i.samples:
                counter += 1
        summary_dict[0] = total_sum / counter
        return summary_dict

    def compute_rand_index(self):
        """
        computing rang index
        :return: rand value
        """
        correct_sum = 0
        sample_counter = 0
        for cluster_i in self.clusters:
            for sample in cluster_i.samples:
                correct_sum += self.correct_counter(sample, cluster_i)
                sample_counter += 1
        correct_sum = correct_sum / 2  # every correct decision is counted for both samples
        sample_counter = sample_counter * (sample_counter - 1) / 2
        return correct_sum / sample_counter

    def correct_counter(self, sample_check, cluster_of):
        """
        rand helper
        :param sample_check: a sample
        :param cluster_of: other cluster
        :return: amount of good predictions with this sample
        """
        counter = 0
        for cluster_i in self.clusters:
            if cluster_i != cluster_of:
                for sample in cluster_i.samples:
                    if sample_check.label != sample.label:
                        counter += 1
        for sample in cluster_of.samples:
            if sample != sample_check:
                if sample_check.label == sample.label:
                    counter += 1
        return counter

    def run(self, max_clusters):
        """
        running the algorithm
        :param max_clusters: wanted number of clusters
        :return: print results
        """
        while len(self.clusters) > max_clusters:
            minimum = -1
            i, j = -1, -1
            final_i, final_j = 0, 0
            for cluster_i in self.clusters:
                i += 1
                j = -1
                for cluster_j in self.clusters:
                    j += 1
                    if cluster_i.c_id == cluster_j.c_id:
                        continue
                    if minimum == -1:
                        minimum = self.link.compute(cluster_i, cluster_j, self.distances_dict)
                        final_i, final_j = i, j
                    if self.link.compute(cluster_i, cluster_j, self.distances_dict) < minimum:
                        minimum = self.link.compute(cluster_i, cluster_j, self.distances_dict)
                        final_i, final_j = i, j
            self.clusters[final_i].merge(self.clusters[final_j])
            del self.clusters[final_j]

        sil_dic = self.compute_summery_silhoeutte(self.distances_dict)
        for cluster_i in self.clusters:
            cluster_i.print_details(sil_dic[cluster_i.c_id])
        print(f"Whole data: silhouette = {round(sil_dic[0],3)}, RI = {round(self.compute_rand_index(),3)}")
