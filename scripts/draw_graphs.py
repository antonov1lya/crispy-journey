import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def draw(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
        print(data_json)
        qps_all = []
        configuration = []
        for file in data_json['data']:
            data = pd.read_csv(file['file'])
            data = data.groupby('recall')['time'].min().reset_index()
            qps = data_json['query'] / (data['time'] / 1e9)
            qps_all.append(qps.to_numpy()[-1])
            configuration.append(file['title'])
            plt.plot(data['recall'], qps,
                     label=file['title'], marker='^', lw=1)
        plt.xticks(np.arange(0.9, 1.0, 0.01))
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1),
            ncol=2
        )
        plt.xlabel('recall@10')
        plt.ylabel('QPS (1/s)')
        plt.grid(True, linestyle=':')
        plt.savefig(data_json['output_file_frontier'],
                    dpi=300, bbox_inches='tight')
        plt.clf()

        print(qps_all)
        for i in range(1, len(qps_all)):
            qps_all[i] /= qps_all[0]
        qps_all[0] = 1

        plt.ylabel('Normalized QPS')

        bars = []
        for i, (conf, qps) in enumerate(zip(configuration, qps_all)):
            bar = plt.bar(i, qps, label=conf)
            bars.append(bar[0])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=9)

        plt.xticks([])
        plt.xlabel('recall@10=0.99')
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1),
            ncol=2
        )
        plt.subplots_adjust(top=0.8)
        plt.savefig(
            data_json['output_file_ablation'],
            dpi=300,
            bbox_inches='tight'
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    draw(args.json_path)


if __name__ == "__main__":
    main()
