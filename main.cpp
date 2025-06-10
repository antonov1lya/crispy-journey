#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "hnsw.h"
#include "hnsw_inference.h"
#include "primitives.h"

// #define SPACE SpaceCosine
#define SPACE SpaceL2

std::string dataset_name = "fashion_mnist";
// std::string dataset_name = "gist";
// std::string dataset_name = "sift";
// std::string dataset_name = "glove";

std::mt19937 gen(0);

std::uniform_real_distribution<float> dist(0, 1);

std::vector<std::vector<FloatType>> query_data;
std::vector<std::set<size_t>> groundtruth_data;

void NormalizeGlove(FloatType* array) {
    for (IntType i = 0; i < 1183514; ++i) {
        Normalize(&(array[i * SIZE]));
    }
}

void ReadData() {
    std::ifstream query(std::string("datasets/") + dataset_name + std::string("/query.txt"));
    std::ifstream groundtruth(std::string("datasets/") + dataset_name +
                              std::string("/groundtruth.txt"));
    int n = 10000;
    if (dataset_name == "gist") {
        n = 1000;
    }
    int k = 10;
    query_data = std::vector<std::vector<FloatType>>(n, std::vector<FloatType>(SIZE));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            query >> query_data[i][j];
        }
        if (dataset_name == "glove") {
            Normalize(&query_data[i][0]);
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

void evaluate(std::ofstream& out, HNSWInference<SPACE>& hnsw, size_t ef, int k = 10) {
    int n = 10000;
    if (dataset_name == "gist") {
        n = 1000;
    }
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
    out << "ef " << ef << "\n";
    out << "avg dist computations " << hnsw.space_.GetComputationsNumber() / n << "\n";
    out << "time " << elapsed_sec.count() << "\n";
}

void Benchmark() {
    // std::ifstream in("indexes/sift.txt");
    // std::ifstream in("reordered/sift/sift_lc.txt");
    // std::ifstream in("reordered/sift/sift_tree.txt");

    // std::ifstream in("reordered/glovek5nn.txt");
    // std::ifstream in("reordered/sift_tree.txt");

    // std::ifstream in("reordered/sift_lc.txt");

    // std::ifstream in("indexes/gist.txt");
    // std::ifstream in("reordered/gist_lc.txt");
    // std::ifstream in("reordered/gist_tree.txt");

    // std::ifstream in("indexes/fashion_mnist.txt");
    std::ifstream in("reordered/fashion_mnist/lc_new.txt");
    // std::ifstream in("reordered/fashion_mnist/lc_new.txt");

    // std::ifstream in("indexes/glove.txt");
    // std::ifstream in("reordered/glove/glove_lc_new.txt");
    // std::ifstream in("reordered/glove/glove_lc_new.txt");
    // std::ifstream in("reordered/glove/glove_lc.txt");
    // std::ifstream in("reordered/glove/glove_tree.txt");
    // std::ifstream in("reordered/glove/glove_gorder.txt");

    // std::ifstream in("reordered/gloveglove_gorder.txt");
    // std::ifstream in("reordered/glove_classic_lc.txt");
    // std::ifstream in("reordered/gloveglove_gorder.txt");
    // std::ifstream in("reordered/glove_tree.txt");
    // std::ifstream in("reordered/glove_new.txt");
    // std::ifstream in("reordered/gloveglovelevel1++.txt");

    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.txt"));
    HNSWInference<SPACE> hnsw(in, in_data);

    if (dataset_name == "glove") {
        for (int i = 0; i < 1183514; ++i) {
            Normalize(hnsw.data + i * SIZE);
        }
    }
    in.close();
    in_data.close();

    ReadData();

    std::cout << "WARMUP\n";
    std::ofstream trash("v3.txt");
    for (int i = 10; i < 100; i += 10) {
        std::cout << i << "\n";
        evaluate(trash, hnsw, i, 10);
    }
    trash.close();

    std::cout << "START\n";
    std::ofstream print("bench/fashion_mnist/lc_new.txt");
    {
        for (int j = 0; j < 5; j++)
            for (int i = 10; i < 51; i += 5) {
                std::cout << i << " " << j << "\n";
                evaluate(print, hnsw, i, 10);
            }
        print << "NEXT\n";
    }
}

// void PrintGraph() {
//     std::ifstream in("sift_reorder/lc_100k.txt");
//     HNSW<SPACE> hnsw(in);
//     in.close();

//     std::ofstream print("output.txt");
//     for (int i = 0; i < hnsw.size_; ++i) {
//         for (int j : hnsw.graph_[i].neighbors_[0]) {
//             print << i << " " << j << "\n";
//         }
//     }
// }

void Reorder() {
    std::ifstream in(std::string("indexes/") + dataset_name + std::string("_classic.txt"));
    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.txt"));
    HNSW<SPACE> hnsw(in, in_data);
    in.close();
    in_data.close();

    // std::ifstream order("perm.txt");
    // for(int i=0; i<hnsw.size_; ++i){
    //     order >> hnsw.reorder_to_new_[i];
    // }

    // for(int i=0; i<hnsw.size_; ++i){
    //     hnsw.reorder_to_old_[hnsw.reorder_to_new_[i]]=i;
    // }

    // std::shuffle(hnsw.reorder_to_new_.begin(), hnsw.reorder_to_new_.end(), gen);
    // for(int i=0; i<hnsw.size_; ++i){
    //     hnsw.reorder_to_old_[hnsw.reorder_to_new_[i]]=i;
    // }

    // int sum=0;
    // for(int i=0; i<hnsw.size_; ++i){
    //     if(hnsw.graph_[i].neighbors_.size()==1){
    //         hnsw.graph_[i].neighbors_[0].clear();
    //     }else{
    //         sum++;
    //     }
    //     // if(i%10000==0){
    //     //     std::cout << i << "\n";
    //     // }
    //     // QueueLess q;
    //     // for(int j: hnsw.graph_[i].neighbors_[0]){
    //     //     q.push({hnsw.space_.Distance(&(hnsw.data_long_[i * SIZE]), &(hnsw.data_long_[j *
    //     SIZE])), j});
    //     // }
    //     // hnsw.graph_[i].neighbors_[0] = hnsw.SelectNeighbours(q, 5, 5);
    // }
    // std::cout << sum << "\n";

    hnsw.ReOrdering();

    std::ofstream out(std::string("reordered/") + dataset_name + std::string("/lc_classic.txt"));
    hnsw.Save(out);
    out.close();
}

void Create() {
    std::ifstream in(std::string("datasets/") + dataset_name + std::string("/data.txt"));
    size_t M, efConstruction, n;
    if (dataset_name == "fashion_mnist") {
        M = 25;
        efConstruction = 600;
        n = 60000;
    }
    if (dataset_name == "sift") {
        M = 25;
        efConstruction = 600;
        n = 1000000;
    }
    if (dataset_name == "glove") {
        M = 25;
        efConstruction = 2500;
        n = 1183514;
    }
    if (dataset_name == "gist") {
        M = 35;
        efConstruction = 800;
        n = 1000000;
    }
    FloatType el = 1.0 / std::log(1.0 * M);
    HNSW<SPACE> hnsw(M, efConstruction, n, in);
    in.close();
    if (dataset_name == "glove") {
        NormalizeGlove(&(hnsw.data_long_[0]));
    }
    for (int i = 0; i < n; ++i) {
        if (i % 1000 == 0) {
            std::cout << i << "\n";
        }
        FloatType level = -std::log(dist(gen)) * el;
        hnsw.Add(level);
    }
    std::ofstream out(std::string("indexes/") + dataset_name + std::string("_classic.txt"));
    hnsw.Save(out);
    out.close();
}

void PrintGraph() {
    // std::ifstream in(std::string("indexes/") + dataset_name + std::string("_classic.txt"));
    // std::ifstream in("reordered/gloveglove_gorder.txt");
    std::ifstream in("indexes/glove.txt");
    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.txt"));
    HNSW<SPACE> hnsw(in, in_data);
    in.close();
    in_data.close();

    std::ofstream out("glove_graph.txt");
    for (int i = 0; i < hnsw.size_; ++i) {
        for (int j : hnsw.graph_[i].neighbors_[0]) {
            out << i << " " << j << "\n";
        }
    }
    out.close();
}

void ReadReorder() {
    std::ifstream in(std::string("indexes/") + dataset_name + std::string(".txt"));
    std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.txt"));
    HNSW<SPACE> hnsw(in, in_data);
    in.close();
    in_data.close();

    std::ifstream order("perm.txt");
    for (int i = 0; i < hnsw.size_; ++i) {
        order >> hnsw.reorder_to_new_[i];
    }

    for (int i = 0; i < hnsw.size_; ++i) {
        hnsw.reorder_to_old_[hnsw.reorder_to_new_[i]] = i;
    }

    // for(int i: hnsw.graph_[0].neighbors_[0]){
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";

    // for(int i: hnsw.graph_[hnsw.reorder_to_old_[0]].neighbors_[0]){
    //     std::cout << hnsw.reorder_to_new_[i] << " ";
    // }
    // std::cout << "\n";

    hnsw.GraphReWrite();

    // std::ofstream out(std::string("reordered/") + dataset_name + std::string("_lc_8196.txt"));
    std::ofstream out(std::string("reordered/") + dataset_name + std::string("/lc_new.txt"));
    hnsw.Save(out);
    out.close();
}

int main() {
    // Create();
    Benchmark();
    // Reorder();
    // PrintGraph();
    // ReadReorder();

    // std::ifstream in("reordered/fashion_mnist/lc_classic.txt");
    // std::ifstream in_data(std::string("datasets/") + dataset_name + std::string("/data.txt"));
    // HNSW<SPACE> hnsw(in, in_data);

    // std::ofstream out("perm.txt");
    // for(int i=0; i<hnsw.size_; ++i){
    //     out << hnsw.reorder_to_new_[i] << " ";
    // }

    return 0;
}
