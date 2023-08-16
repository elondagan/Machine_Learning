import sys
import data
import distances
import link
import AgglomerativeClustering


def main(argv):
    path = argv[1]
    data_set_dict = data.Data(path)
    samples_list = data_set_dict.create_samples()
    distances_dict = distances.create_distances_dict(samples_list)
    print("single link:")
    single = link.SingleLink()
    ag = AgglomerativeClustering.AgglomerativeClustering(single, samples_list, distances_dict)
    ag.run(7)
    print()
    print("complete link")
    complete = link.CompleteLink()
    ag = AgglomerativeClustering.AgglomerativeClustering(complete, samples_list, distances_dict)
    ag.run(7)


if __name__ == '__main__':
    main(sys.argv)
