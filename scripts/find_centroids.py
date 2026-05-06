import argparse

import h5py
import numpy as np
from sklearn.cluster import KMeans

NBITS = 8


def find_centroid(input_path: str, M: int, angular: bool):
    with h5py.File(input_path, "r") as file:
        train_vectors = file["train"][:]
    if angular:
        train_vectors /= np.linalg.norm(train_vectors, axis=1, keepdims=True)
    N, D = train_vectors.shape

    kmeans = KMeans(n_clusters=M, random_state=42).fit(train_vectors)

    medoid_indices = []
    for i in range(M):
        mask = kmeans.labels_ == i
        if np.any(mask):
            cluster_points = train_vectors[mask]
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            medoid_indices.append(np.where(mask)[0][np.argmin(distances)])

    for idx in medoid_indices:
        print(idx)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--angular", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    find_centroid(args.input_file, args.M, args.angular)


if __name__ == "__main__":
    main()
