import h5py
import argparse
import sys
import os


def convert_hdf5(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(input_path, "r") as f:
        f["train"][:].tofile(os.path.join(output_dir, "data.bin"))
        f["test"][:].tofile(os.path.join(output_dir, "query.bin"))
        f["neighbors"][:].tofile(os.path.join(output_dir, "groundtruth.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to HDF5 file")
    parser.add_argument(
        "-o", "--output-dir", default=".", help="Output directory (default: current)"
    )
    args = parser.parse_args()
    convert_hdf5(args.input, args.output_dir)


if __name__ == "__main__":
    main()
