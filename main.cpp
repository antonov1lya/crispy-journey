#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "hnsw.h"
#include "primitives.h"

std::mt19937 gen(0);

std::uniform_real_distribution<FloatType> dist(0, 1);

void evaluate(std::ofstream& out, HNSW<SpaceL2>& hnsw, size_t ef, int k = 10) {
    std::ifstream query("sift_query.txt");
    std::ifstream groundtruth("sift_groundtruth.txt");

    int n = 10000;
    double result = 0;
    hnsw.space_.FlushComputationsNumber();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        Point v(128);
        for (int j = 0; j < 128; ++j) {
            query >> v[j];
        }
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

void PrintGraph(HNSW<SpaceL2>& hnsw){
    std::ofstream print("output.txt");
    for(int i=0; i<hnsw.size_; ++i){
        for(int j: hnsw.graph_[i].neighbors_[0]){
            print << i << " " << j << "\n";
        }
    }
}

int main() {
    std::ifstream in("reordered_opt.txt");
    HNSW<SpaceL2> hnsw(in);
    in.close();

    // PrintGraph(hnsw);
    // return 0;

    // std::ofstream out("output4.txt");
    // // hnsw.Save(out);

    // for(int i=0; i<1e6; ++i){
    //     for(int j: hnsw.graph_[i].neighbors_[0]){
    //         out << i << " " << j << "\n";
    //     }
    // }

    // hnsw.ReOrdering();

    // // // // // // // hnsw.MemoryManager();

    // std::ofstream out("reordered_opt.txt");
    // hnsw.Save(out);
    // out.close();

    std::cout << "WARMUP\n";
    std::ofstream trash("bench/trash.txt");
    for (int i = 10; i < 51; i += 5) {
        evaluate(trash, hnsw, i, 10);
    }
    trash.close();

    std::cout << "START\n";
    std::ofstream print("bench/new.txt");
    {
        for (int j = 0; j < 5; j++)
            for (int i = 10; i < 51; i += 5) {
                std::cout << i << " " << j << "\n";
                evaluate(print, hnsw, i, 10);
            }
        print << "NEXT\n";
    }

    return 0;
}
