#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "hnsw.h"
#include "hnsw_inference.h"
#include "primitives.h"

std::string dataset_name = DATASET;

std::mt19937 gen(0);
std::uniform_real_distribution<FloatType> dist(0, 1);

FloatType* query_data;
std::vector<std::set<IntType>> groundtruth_data;

const int kNN = 10;

void ReadData() {
    std::ifstream query(std::string("datasets/") + dataset_name + std::string("/query.bin"),
                        std::ios::binary);
    std::ifstream groundtruth(
        std::string("datasets/") + dataset_name + std::string("/groundtruth.bin"),
        std::ios::binary);
    int n = 10000;
    if (dataset_name == "gist") {
        n = 1000;
    }

    query_data = (FloatType*)malloc(n * SIZE * sizeof(FloatType));
    query.read(reinterpret_cast<char*>(query_data), n * SIZE * sizeof(FloatType));

    auto ReadBinaryInt = [&groundtruth](IntType& value) {
        groundtruth.read(reinterpret_cast<char*>(&value), sizeof(IntType));
    };

    groundtruth_data = std::vector<std::set<IntType>>(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 100; ++j) {
            IntType x;
            ReadBinaryInt(x);
            if (j < kNN) {
                groundtruth_data[i].insert(x);
            }
        }
    }
    query.close();
    groundtruth.close();
}

void evaluate(std::ofstream& out, HNSWInference<SPACE>& hnsw, size_t ef) {
    int n = 10000;
    if (dataset_name == "gist1m") {
        n = 1000;
    }
    double result = 0;
    hnsw.space_.FlushComputationsNumber();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        auto res = hnsw.Search(&query_data[i * SIZE], kNN, ef);
        double count = 0;
        for (auto x : res) {
            count += groundtruth_data[i].count(x);
        }
        count /= kNN;
        result += count;
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    out << "recall " << result / n << "\n";
    out << "ef " << ef << "\n";
    out << "avg dist computations " << hnsw.space_.GetComputationsNumber() / n << "\n";
    out << "time " << elapsed_sec.count() << "\n";
}

void Benchmark() {
    std::ifstream in(std::string("indexes/") + dataset_name + std::string("/base.bin"),
                     std::ios::binary);
    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.bin"),
                          std::ios::binary);
    HNSWInference<SPACE> hnsw(in, in_data);
    in.close();
    in_data.close();

    std::ifstream file_data_pq(
        std::string("datasets/") + dataset_name + std::string("/data_pq32.bin"), std::ios::binary);
    std::ifstream file_centroids(
        std::string("datasets/") + dataset_name + std::string("/centroids32.bin"),
        std::ios::binary);
    std::ifstream file_matrix(
        std::string("datasets/") + dataset_name + std::string("/matrix32.bin"), std::ios::binary);
    hnsw.LoadPQ(file_data_pq, file_centroids);
    hnsw.LoadPQMatrix(file_matrix);
    file_data_pq.close();
    file_centroids.close();
    file_matrix.close();

    // std::ifstream file_matrix_rerank(
    //     std::string("datasets/") + dataset_name + std::string("/rerank_matrix.bin"),
    //     std::ios::binary);
    // hnsw.LoadReRankMatrix(file_matrix_rerank);
    // file_matrix_rerank.close();

    ReadData();

    std::cout << "WARMUP\n";
    std::ofstream print(std::string("logs/") + dataset_name + std::string("/res.txt"));
    for (int i = 400; i <= 400; i += 1000) {
        std::cout << i << "\n";
        evaluate(print, hnsw, i);
    }
    print.close();
}

void Reorder() {
    std::ifstream in(std::string("indexes/") + dataset_name + std::string("/base.bin"),
                     std::ios::binary);
    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.bin"),
                          std::ios::binary);
    HNSW<SPACE> hnsw(in, in_data);
    in.close();
    in_data.close();

    hnsw.ReOrdering();

    std::ofstream out(std::string("indexes/") + dataset_name + std::string("/reordering_bfs.bin"),
                      std::ios::binary);
    hnsw.Save(out);
    out.close();
}

void SSG() {
    std::ifstream in(std::string("indexes/") + dataset_name + std::string("/classic.bin"),
                     std::ios::binary);
    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.bin"),
                          std::ios::binary);
    HNSW<SPACE> hnsw(in, in_data);
    in.close();
    in_data.close();

    for (IntType _ = 0; _ < 5; ++_) {
        hnsw.ImproveSSG();
    }

    std::ofstream out(std::string("indexes/") + dataset_name + std::string("/ssg_classic.bin"),
                      std::ios::binary);
    hnsw.Save(out);
    out.close();
}

void Create() {
    std::ifstream in(std::string("datasets/") + dataset_name + std::string("/data.bin"),
                     std::ios::binary);
    size_t M, efConstruction, n;
    if (dataset_name == "fashion_mnist") {
        M = 25;
        efConstruction = 600;
        n = 60000;
    }
    if (dataset_name == "sift1m") {
        M = 25;
        efConstruction = 600;
        n = 1000000;
    }
    if (dataset_name == "glove100") {
        M = 25;
        efConstruction = 2500;
        n = 1183514;
    }
    if (dataset_name == "deep1b") {
        M = 25;
        efConstruction = 2500;
        n = 9990000;
    }
    if (dataset_name == "gist1m") {
        M = 35;
        efConstruction = 800;
        n = 1000000;
    }
    FloatType el = 1.0 / std::log(1.0 * M);
    HNSW<SPACE> hnsw(M, efConstruction, n, in);
    in.close();
    for (int i = 0; i < n; ++i) {
        if (i % 1000 == 0) {
            std::cout << i << "\n";
        }
        FloatType level = -std::log(dist(gen)) * el;
        hnsw.Add(level);
    }
    std::ofstream out(std::string("indexes/") + dataset_name + std::string("/base.bin"),
                      std::ios::binary);
    hnsw.Save(out);
    out.close();
}

int main() {
    // DO: sudo sysctl vm.nr_hugepages=1024
    // Create();
    Benchmark();
    // Reorder();
    // PrintGraph();
    // ReadReorder();
    // SSG();

    return 0;
}
