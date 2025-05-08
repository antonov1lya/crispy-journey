#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "hnsw.h"
#include "primitives.h"

#define SPACE SpaceCosine

std::mt19937 gen(0);

std::uniform_real_distribution<FloatType> dist(0, 1);

void evaluate(std::ofstream& out, HNSW<SPACE>& hnsw, size_t ef, int k = 10) {
    std::ifstream query("datasets/glove/query.txt");
    std::ifstream groundtruth("datasets/glove/groundtruth.txt");

    int n = 10000;
    double result = 0;
    hnsw.space_.FlushComputationsNumber();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        Point v(SIZE);
        for (int j = 0; j < SIZE; ++j) {
            query >> v[j];
        }
        v.Normalize();
        auto res = hnsw.Search(v, k, ef);
        std::vector<size_t> w(100);
        for (int j = 0; j < 100; ++j) {
            groundtruth >> w[j];
        }
        std::set<size_t> s;
        double count = 0;
        for (int j = 0; j < k; ++j) {
            s.insert(w[j]);
        }
        for (auto x : res) {
            count += s.count(x);
        }
        count /= k;
        result += count;
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    out << "recall " << result / n << "\n";
    out << "avg dist computations " << hnsw.space_.GetComputationsNumber() / n << "\n";
    out << "time " << elapsed_sec.count() << "\n";
    query.close();
    groundtruth.close();
}

void PrintGraph(HNSW<SPACE>& hnsw) {
    std::ofstream print("output.txt");
    for (int i = 0; i < hnsw.size_; ++i) {
        for (int j : hnsw.graph_[i].neighbors_[0]) {
            print << i << " " << j << "\n";
        }
    }
}

void Benchmark() {
    std::ifstream in("indexes/glove.txt");
    HNSW<SPACE> hnsw(in);
    in.close();

    std::cout << "WARMUP\n";
    std::ofstream trash("bench/trash.txt");
    for (int i = 10; i < 101; i += 10) {
        evaluate(trash, hnsw, i, 10);
    }
    trash.close();

    std::cout << "START\n";
    std::ofstream print("bench/test_glove.txt");
    {
        for (int j = 0; j < 5; j++)
            for (int i = 10; i < 201; i += 10) {
                std::cout << i << " " << j << "\n";
                evaluate(print, hnsw, i, 10);
            }
        print << "NEXT\n";
    }
}

void Reorder() {
    std::ifstream in("indexes/glove.txt");
    HNSW<SPACE> hnsw(in);
    in.close();

    hnsw.ReOrdering();

    std::ofstream out("reordered_glove_lc_new.txt");
    hnsw.Save(out);
    out.close();
}

void Create() {
    std::ifstream in("datasets/glove/data.txt");
    // size_t M = 25, efConstruction = 600, n = 1000000;
    size_t M = 25, efConstruction = 2500, n = 1183514;
    FloatType el = 1.0 / std::log(1.0 * M);
    HNSW<SPACE> hnsw(M, efConstruction, n);
    for (int i = 0; i < n; ++i) {
        if (i % 1000 == 0) {
            std::cout << i << "\n";
        }
        Point point(SIZE);
        for (int j = 0; j < SIZE; ++j) {
            in >> point[j];
        }
        point.Normalize();
        int level = static_cast<int>(-std::log(dist(gen)) * el);
        hnsw.Add(point, level);
    }
    std::ofstream out("indexes/glove.txt");
    hnsw.Save(out);
    out.close();
}

int main() {
    // Create();
    Benchmark();
    // Reorder();

    return 0;
}
