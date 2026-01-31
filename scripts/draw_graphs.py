import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_file(file):
    x = []
    y = []
    with open(file, 'r') as f:
        for i in f:
            arr = i.split()
            if arr[0] == 'recall':
                x.append(float(arr[1]))
            if arr[0] == 'time':
                y.append(1000/(float(arr[1])/1e9))
    return x, y


def draw(input_dir: str, output_file: str):
    directory = Path(input_dir)
    txt_files = list(directory.glob("*.txt"))
    for file in txt_files:
        x, y = read_file(file)
        plt.plot(x, y, label=file.stem, marker='^')
    plt.xticks(np.arange(0.9, 1.0, 0.01))
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('qps')
    plt.grid(True, axis='y')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    draw(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
