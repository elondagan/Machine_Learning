def create_distances_dict(samples_list):
    """
    creating distances dict, while key holds both the sample id`s and value is dist
    :param samples_list: list of Samples
    :return: dict of distances
    """
    distance_dict = {}
    for i in range(len(samples_list)):
        for j in range(i+1, len(samples_list)):
            distance = samples_list[i].compute_euclidean_distance(samples_list[j])
            distance_dict[(samples_list[i].s_id, samples_list[j].s_id)] = distance
            distance_dict[(samples_list[j].s_id, samples_list[i].s_id)] = distance

    return distance_dict

