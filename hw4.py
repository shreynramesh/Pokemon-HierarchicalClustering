import csv
import heapq
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath):
    """
    This function takes one parameter, the filepath, and converts that csv file to a list of dictionaries that
    specifies the attributes of each Pokemon
    :param filepath: a CSV file containing Pokemon
    :return: list of dictionaries where the column headers are keys and the row values are the values
    """

    data = {}

    # Opening CSV file to read
    try:
        with open(filepath, 'r') as file:
            dict_reader = csv.DictReader(file)
            data = list(dict_reader)  # Converting DictReader to a list of dictionaries
    except (FileNotFoundError, IOError):
        print("Invalid file name. Could not open")

    return data


def calc_features(row):
    """
    This function takes as input the dict representing one Pokemon, and computes
    the feature representation (x1, . . . , x6). Specifically,
        1. x1 = Attack
        2. x2 = Sp. Attack
        3. x3 = Speed
        4. x4 = Defense
        5. x5 = Sp. Def
        6. x6 = HP
    :param row: a dictionary representing one Pokemon
    :return: numpy array of shape (6,) and dtype int64. The first element is x1 and so on
            with the sixth element being x6.
    """
    x1 = row["Attack"]
    x2 = row["Sp. Atk"]
    x3 = row["Speed"]
    x4 = row["Defense"]
    x5 = row["Sp. Def"]
    x6 = row["HP"]
    feature = np.array([x1, x2, x3, x4, x5, x6], dtype=np.int64)

    return feature


def hac(features):
    """
    - This function mimics the behavior of SciPy's HAC function linkage(). Using complete linkage, it performs
    hierarchical agglomerative clustering.
    :param features: a list of n numpy arrays of shape (6,), where each array is a feature representation of a Pokemon
    :return: a numpy array of shape (n-1) x 4 where for any i, Z[i, 0] and Z[i, 1] represent the indices of the two
            clusters that were merged in the ith iteration of the clustering algorithm. Z[i, 2] will be the complete
            linkage distance between the two clusters. Lastly, Z[i, 3] is the size of the new cluster formed by the
            merge.
    """

    num_pokemon = len(features)
    dist_matrix = np.zeros((num_pokemon, num_pokemon))
    heap = []  # Stores sorted distances
    cluster_availability_and_size = {}

    # Filling the dist_matrix with the distances of the original features and pushing those distances onto a min-heap
    # The heap stores 3-tuples in the form of (distance, feature1, feature2) where feature1 < feature2
    for i, _ in enumerate(dist_matrix):
        cluster_availability_and_size[i] = 1
        for j, _ in enumerate(dist_matrix[i]):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(features[i] - features[j])
                heap_entry = (dist_matrix[i, j], min(i, j), max(i, j))
                heapq.heappush(heap, heap_entry)

    z = np.zeros((num_pokemon - 1, 4))  # Matrix we return

    for i in range(len(z)):
        clusters_to_merge = heapq.heappop(heap)

        # Re-popping if either cluster is not available
        while clusters_to_merge[1] not in cluster_availability_and_size \
                or clusters_to_merge[2] not in cluster_availability_and_size:
            clusters_to_merge = heapq.heappop(heap)

        z[i, 0] = clusters_to_merge[1]  # cluster1 to merge
        z[i, 1] = clusters_to_merge[2]  # cluster2 to merge
        z[i, 2] = clusters_to_merge[0]  # distance between cluster1 and cluster2
        # Adding the size of the 2 clusters that are merged
        z[i, 3] = cluster_availability_and_size[z[i, 0]] + cluster_availability_and_size[z[i, 1]]

        # Creating an extra row and column in dist_matrix to make room for new cluster
        new_column = np.zeros((len(dist_matrix), 1))
        new_row = np.zeros((1, len(dist_matrix) + 1))
        for j in range(len(new_column)):
            new_column[j] = max(dist_matrix[int(z[i, 1]), j], dist_matrix[int(z[i, 0]), j])
            new_row[0, j] = new_column[j]
            heapq.heappush(heap, (new_column[j], min(num_pokemon + i, j), max(num_pokemon + i, j)))

        dist_matrix = np.append(dist_matrix, new_column, axis=1)
        dist_matrix = np.append(dist_matrix, new_row, axis=0)

        # Updating cluster availability and size
        del cluster_availability_and_size[z[i, 0]]
        del cluster_availability_and_size[z[i, 1]]
        cluster_availability_and_size[num_pokemon + i] = z[i, 3]

    return z


def imshow_hac(z, names):
    """
    This function displays a graph that visualizes hierarchical clustering.

    :param z: numpy array output from hac()
    :param names: list of String names that correspond to Pokemon with size n x 1
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(z, labels=names, leaf_rotation=90, ax=ax)

    # adjust the layout to make sure x labels are not cut off
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    features_and_names = [(calc_features(row), row["Name"]) for row in load_data("Pokemon.csv")[:50]]
    Z = hac([row[0] for row in features_and_names])
    names = [row[1] for row in features_and_names]
    imshow_hac(Z, names)