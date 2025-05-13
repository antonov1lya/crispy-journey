#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "hnsw.h"
#include "primitives.h"

// #define SPACE SpaceCosine
#define SPACE SpaceL2

std::mt19937 gen(0);

std::uniform_real_distribution<FloatType> dist(0, 1);

std::vector<std::vector<FloatType>> query_data;
std::vector<std::set<size_t>> groundtruth_data;

void ReadData() {
    std::ifstream query("datasets/fashion_mnist/query.txt");
    std::ifstream groundtruth("datasets/fashion_mnist/groundtruth.txt");
    int n = 10000;
    int k = 10;
    query_data = std::vector<std::vector<FloatType>>(n, std::vector<FloatType>(SIZE));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            query >> query_data[i][j];
        }
    }
    groundtruth_data = std::vector<std::set<size_t>>(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 100; ++j) {
            size_t x;
            groundtruth >> x;
            if (j < k) {
                groundtruth_data[i].insert(x);
            }
        }
    }
    query.close();
    groundtruth.close();
}

void evaluate(std::ofstream& out, HNSW<SPACE>& hnsw, size_t ef, int k = 10) {
    int n = 10000;
    double result = 0;
    hnsw.space_.FlushComputationsNumber();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        auto res = hnsw.Search(&query_data[i][0], k, ef);
        double count = 0;
        for (auto x : res) {
            count += groundtruth_data[i].count(x);
        }
        count /= k;
        result += count;
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    out << "recall " << result / n << "\n";
    out << "avg dist computations " << hnsw.space_.GetComputationsNumber() / n << "\n";
    out << "time " << elapsed_sec.count() << "\n";
}

void PrintGraph() {
    std::ifstream in("sift_random.txt");
    HNSW<SPACE> hnsw(in);
    in.close();

    std::ofstream print("output.txt");
    for (int i = 0; i < hnsw.size_; ++i) {
        for (int j : hnsw.graph_[i].neighbors_[0]) {
            print << i << " " << j << "\n";
        }
    }
}

void Benchmark() {
    std::ifstream in("trash1.txt");
    HNSW<SPACE> hnsw(in);
    in.close();

    ReadData();

    std::cout << "WARMUP\n";
    std::ofstream trash("bench/trash.txt");
    for (int i = 10; i < 101; i += 10) {
        evaluate(trash, hnsw, i, 10);
    }
    trash.close();

    // std::cout << "START\n";
    // std::ofstream print("big_trash.txt");
    // {
    //     for (int j = 0; j < 5; j++)
    //         for (int i = 10; i < 201; i += 10) {
    //             std::cout << i << " " << j << "\n";
    //             evaluate(print, hnsw, i, 10);
    //         }
    //     print << "NEXT\n";
    // }
}

void Reorder() {
    std::ifstream in("trash.txt");
    HNSW<SPACE> hnsw(in);
    in.close();

    hnsw.ReOrdering();

    std::ofstream out("trash1.txt");
    hnsw.Save(out);
    out.close();
}

void Create() {
    std::ifstream in("datasets/fashion_mnist/data.txt");
    // size_t M = 25, efConstruction = 600, n = 1000000;
    // size_t M = 25, efConstruction = 2500, n = 1183514;
    size_t M = 25, efConstruction = 600, n = 60000;
    FloatType el = 1.0 / std::log(1.0 * M);
    HNSW<SPACE> hnsw(M, efConstruction, n);
    for (int i = 0; i < n; ++i) {
        if (i % 1000 == 0) {
            std::cout << i << "\n";
        }
        // Point point(SIZE);
        for (int j = 0; j < SIZE; ++j) {
            // in >> point[j];
            in >> hnsw.data_long_[i * SIZE + j];
        }
        // point.Normalize();
        int level = static_cast<int>(-std::log(dist(gen)) * el);
        hnsw.Add(level);
    }
    // for(int i=0; i<10; ++i){
    //     for(int j=0; j<SIZE; ++j){
    //         std::cout << hnsw.data_long_[i * SIZE + j] << " ";
    //     }
    // }
    std::ofstream out("trash.txt");
    hnsw.Save(out);
    out.close();
}

// void ReSave() {
//     std::ifstream in("glove_heuristic.txt");
//     HNSW<SPACE> hnsw(in);

//     size_t M = 25, efConstruction = 2500, n = 1183514;

//     std::vector<int> reorder_to_new_(n);
//     for (int i = 0; i < n; i++) {
//         reorder_to_new_[hnsw.reorder_to_old_[i]] = i;
//     }

//     std::ifstream dt("datasets/glove/data.txt");
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < SIZE; j++) {
//             dt >> hnsw.data_[reorder_to_new_[i]][j];
//         }
//         hnsw.data_[reorder_to_new_[i]].Normalize();
//     }

//     std::ofstream out("glove_heuristic_pres6.txt");
//     hnsw.Save(out);
//     out.close();
// }

int main() {
    // Create();
    Benchmark();
    // Reorder();
    // ReSave();
    // PrintGraph();

    return 0;
}
