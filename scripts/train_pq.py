import argparse
import h5py
import numpy as np
import faiss

NBITS = 8


def train_pq(input_path: str, output_dir: str, M: int, angular: bool):
    with h5py.File(input_path, "r") as file:
        train_vectors = file["train"][:]
    if angular:
        train_vectors /= np.linalg.norm(train_vectors, axis=1, keepdims=True)
    N, D = train_vectors.shape

    opq = faiss.OPQMatrix(D, M)
    pq = faiss.IndexPQ(D, M, NBITS)
    index = faiss.IndexPreTransform(opq, pq)
    index.train(train_vectors)
    index.add(train_vectors)

    identity_matrix = np.eye(D, dtype='float32')
    transformed_matrix = opq.apply_py(identity_matrix).T
    PQ = pq.pq
    centroids = faiss.vector_to_array(PQ.centroids)

    print(transformed_matrix[0])
    transformed_matrix.tofile(f'{output_dir}/matrix{M}.bin')
    centroids.tofile(f'{output_dir}/centroids{M}.bin')
    faiss.vector_to_array(pq.codes).tofile(f'{output_dir}/data_pq{M}.bin')

    print('l2^2 norm diff:')
    for i in range(16):
        before = np.linalg.norm(train_vectors[0] - train_vectors[i]) ** 2
        after = np.linalg.norm(train_vectors[0] - index.reconstruct(i)) ** 2
        print(f'original: {before: .3f}, quantized: {after: .3f}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--M', type=int, required=True)
    parser.add_argument('--angular', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    train_pq(args.input_file, args.output_dir, args.M, args.angular)


if __name__ == "__main__":
    main()
