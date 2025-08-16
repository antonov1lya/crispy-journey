#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <vector>

#include "primitives.h"

struct Node {
    Node(IntType max_level) {
        neighbors_.resize(max_level + 1);
    }
    std::vector<std::vector<IntType>> neighbors_;
};

template <typename Space>
struct HNSW {
    HNSW(IntType M, IntType ef_construction, IntType max_elements, std::ifstream& file_data)
        : M_{M},
          maxM_{M},
          maxM0_{2 * M},
          ef_construction_{ef_construction},
          max_elements_{max_elements} {
        was_ = std::vector<IntType>(max_elements_);
        graph_.reserve(max_elements_);
        data_long_ = static_cast<FloatType*>(aligned_alloc(64, (max_elements * SIZE) * sizeof(FloatType)));
        reorder_to_new_.resize(max_elements);
        reorder_to_old_.resize(max_elements);
        for (int i = 0; i < max_elements; ++i) {
            reorder_to_new_[i] = i;
            reorder_to_old_[i] = i;
        }
        for (IntType node = 0; node < max_elements; ++node) {
            for (IntType i = 0; i < SIZE; ++i) {
                FloatType x;
                file_data >> x;
                data_long_[reorder_to_new_[node] * SIZE + i] = x;
            }
        }
    }
    HNSW(std::ifstream& file, std::ifstream& file_data);
    ~HNSW() {
        free(data_long_);
    }
    void Add(int level);
    QueueLess SearchLayer(FloatType* query, IntType enter_point, IntType ef, IntType level);
    std::vector<IntType> SelectNeighbours(QueueLess& candidates, IntType M, IntType maxM);
    std::vector<IntType> Search(FloatType* query, IntType K, IntType ef);
    std::vector<IntType> SSG(IntType node, std::vector<IntType>& candidates, IntType M);
    void BFSTreeReOrdering();
    void MSTTreeReOrdering();
    // void SumOfModulesReOrdering();
    void GraphReWrite();
    void ReOrdering();
    IntType DfsStat(IntType cur);
    void DfsReorder(IntType cur);

    void Improve();
    void Save(std::ofstream& file);
    IntType M_;
    IntType maxM_;
    IntType maxM0_;
    IntType ef_construction_;
    IntType enter_point_;
    IntType size_ = 0;
    IntType current_was_ = 0;
    IntType max_elements_;
    int max_level_ = -1;
    FloatType* data_long_;
    std::vector<IntType> was_;
    std::vector<Node> graph_;

    std::vector<std::vector<IntType>> bfs_tree_;
    std::vector<IntType> dfs_stat_;
    std::vector<IntType> reorder_to_new_;
    std::vector<IntType> reorder_to_old_;
    IntType reorder_num = 0;

    IntType size0_ = 0;

    Space space_;
    FloatType ssg_cos = 0.5;
};

template <typename Space>
void HNSW<Space>::Add(int level) {
    graph_.push_back(Node(level));
    IntType index = size_++;
    IntType enter_point = enter_point_;
    for (int i = max_level_; i > level; --i) {
        enter_point = SearchLayer(&(data_long_[index * SIZE]), enter_point, 1, i).top().second;
    }
    for (int i = std::min(max_level_, level); i >= 0; --i) {
        IntType maxM = maxM_;
        if (i == 0) {
            maxM = maxM0_;
        }
        auto nearest_neighbors =
            SearchLayer(&(data_long_[index * SIZE]), enter_point, ef_construction_, i);
        graph_[index].neighbors_[i] = SelectNeighbours(nearest_neighbors, M_, maxM);
        enter_point = graph_[index].neighbors_[i][0];
        for (IntType next : graph_[index].neighbors_[i]) {
            graph_[next].neighbors_[i].push_back(index);
            if (graph_[next].neighbors_[i].size() > maxM) {
                QueueLess queue;
                for (IntType neighbour : graph_[next].neighbors_[i]) {
                    queue.emplace(space_.Distance(&(data_long_[next * SIZE]),
                                                  &(data_long_[neighbour * SIZE])),
                                  neighbour);
                }
                graph_[next].neighbors_[i] = SelectNeighbours(queue, maxM, maxM);
            }
        }
    }
    if (level > max_level_) {
        max_level_ = level;
        enter_point_ = index;
    }
}

template <typename Space>
inline QueueLess HNSW<Space>::SearchLayer(FloatType* query, IntType enter_point, IntType ef,
                                          IntType level) {
    was_[enter_point] = ++current_was_;
    QueueGreater candidates;
    FloatType enter_point_distance = space_.Distance(&(data_long_[enter_point * SIZE]), query);
    candidates.emplace(enter_point_distance, enter_point);
    QueueLess nearest_neighbours;
    nearest_neighbours.emplace(enter_point_distance, enter_point);
    while (!candidates.empty()) {
        IntType current = candidates.top().second;
        candidates.pop();
        FloatType furthest_distance = nearest_neighbours.top().first;
        if (space_.Distance(&(data_long_[current * SIZE]), query) > furthest_distance and
            nearest_neighbours.size() == ef) {
            break;
        }
        const IntType cursz = graph_[current].neighbors_[level].size();
        for (int i = 0; i < cursz; ++i) {
            IntType next = graph_[current].neighbors_[level][i];
            if (was_[next] != current_was_) {
                was_[next] = current_was_;
                furthest_distance = nearest_neighbours.top().first;
                FloatType distance = space_.Distance(&(data_long_[next * SIZE]), query);
                if (distance < furthest_distance or nearest_neighbours.size() < ef) {
                    candidates.emplace(distance, next);
                    nearest_neighbours.emplace(distance, next);
                    if (nearest_neighbours.size() > ef) {
                        nearest_neighbours.pop();
                    }
                }
            }
        }
    }
    return nearest_neighbours;
}

template <typename Space>
inline std::vector<IntType> HNSW<Space>::SelectNeighbours(QueueLess& candidates, IntType M,
                                                          IntType maxM) {
    bool simple = false;
    if (simple) {
        std::vector<IntType> array;
        array.reserve(maxM + 1);
        while (!candidates.empty()) {
            array.push_back(candidates.top().second);
            candidates.pop();
        }
        std::reverse(array.begin(), array.end());
        array.resize(std::min(M, static_cast<IntType>(array.size())));
        return array;
    }
    if (candidates.size() < M) {
        std::vector<IntType> array;
        array.reserve(maxM + 1);
        while (!candidates.empty()) {
            array.push_back(candidates.top().second);
            candidates.pop();
        }
        std::reverse(array.begin(), array.end());
        return array;
    }
    std::vector<std::pair<FloatType, IntType>> queue;
    queue.reserve(candidates.size());
    while (!candidates.empty()) {
        queue.push_back(candidates.top());
        candidates.pop();
    }
    std::reverse(queue.begin(), queue.end());
    std::vector<IntType> selected;
    selected.reserve(maxM + 1);
    for (auto& [distance, element] : queue) {
        if (selected.size() >= M) {
            break;
        }
        bool good = true;
        for (IntType neighbour : selected) {
            FloatType cur_distance =
                space_.Distance(&(data_long_[element * SIZE]), &(data_long_[neighbour * SIZE]));
            if (cur_distance < distance) {
                good = false;
            }
        }
        if (good) {
            selected.push_back(element);
        }
    }
    return selected;
}

template <typename Space>
inline void HNSW<Space>::BFSTreeReOrdering() {
    std::vector<FloatType> means1(SIZE);
    for (int i = 0; i < size_; ++i) {
        for (int j = 0; j < means1.size(); ++j) {
            means1[j] += data_long_[i * SIZE + j];
        }
    }
    for (int j = 0; j < means1.size(); ++j) {
        means1[j] /= size_;
    }
    FloatType best_dist = 1e9, best_i = -1;
    for (int i = 0; i < size_; ++i) {
        if (space_.Distance(&(data_long_[i * SIZE]), &(means1[0])) < best_dist) {
            best_dist = space_.Distance(&(data_long_[i * SIZE]), &(means1[0]));
            best_i = i;
        }
    }

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0, 1);

    std::queue<IntType> queue;
    bfs_tree_.resize(size_);
    std::vector<bool> was(size_);
    was[best_i] = 1;
    queue.push(best_i);
    int cnt = 0;
    while (!queue.empty()) {
        ++cnt;
        int cur = queue.front();
        queue.pop();
        for (auto next : graph_[cur].neighbors_[0]) {
            if(dist(gen)>=0.9) continue;
            if (!was[next]) {
                was[next] = 1;
                queue.push(next);
                bfs_tree_[cur].push_back(next);
            }
        }
    }
    dfs_stat_ = std::vector<IntType>(size_);
    DfsStat(best_i);
    reorder_to_new_ = std::vector<IntType>(size_);
    reorder_to_old_ = std::vector<IntType>(size_);
    DfsReorder(best_i);
}

template <typename Space>
inline void HNSW<Space>::MSTTreeReOrdering() {
    std::vector<FloatType> means1(SIZE);
    for (int i = 0; i < size_; ++i) {
        for (int j = 0; j < means1.size(); ++j) {
            means1[j] += data_long_[i * SIZE + j];
        }
    }
    for (int j = 0; j < means1.size(); ++j) {
        means1[j] /= size_;
    }
    FloatType best_dist = 1e9, best_i = -1;
    for (int i = 0; i < size_; ++i) {
        if (space_.Distance(&(data_long_[i * SIZE]), &(means1[0])) < best_dist) {
            best_dist = space_.Distance(&(data_long_[i * SIZE]), &(means1[0]));
            best_i = i;
        }
    }

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> distt(0, 1);

    std::vector<std::vector<int>> full(size_);
    for (int i = 0; i < size_; ++i) {
        for (auto next : graph_[i].neighbors_[0]) {
            full[i].push_back(next);
            full[next].push_back(i);
        }
    }

    for (int i = 0; i < size_; ++i) {
        std::sort(full[i].begin(), full[i].end());
        auto last = std::unique(full[i].begin(), full[i].end());
        full[i].resize(std::distance(full[i].begin(), last));
    }

    std::set<std::pair<FloatType, int64_t>> q;
    std::vector<FloatType> min_e(size_, 1e18);
    std::vector<int64_t> sel_e(size_, -1);
    q.insert(std::make_pair(0, best_i));
    min_e[best_i] = 0;
    std::vector<bool> selected(size_, false);
    while (!q.empty()) {
        int64_t v = q.begin()->second;
        selected[v] = true;
        q.erase(q.begin());
        for (int64_t to : full[v]) {
            FloatType cost = space_.Distance(&(data_long_[v * SIZE]), &(data_long_[to * SIZE]));
            if (!selected[to] and cost < min_e[to]) {
                q.erase(std::make_pair(min_e[to], to));
                min_e[to] = cost;
                sel_e[to] = v;
                q.insert(std::make_pair(min_e[to], to));
            }
        }
    }
    bfs_tree_.resize(size_);
    for (int i = 0; i < size_; ++i) {
        if (sel_e[i] != -1) {
            bfs_tree_[sel_e[i]].push_back(i);
        }
    }
    
    dfs_stat_ = std::vector<IntType>(size_);
    DfsStat(best_i);
    reorder_to_new_ = std::vector<IntType>(size_);
    reorder_to_old_ = std::vector<IntType>(size_);
    DfsReorder(best_i);
}


template <typename Space>
inline void HNSW<Space>::GraphReWrite() {
    std::vector<Node> graph_new_;
    for (IntType i = 0; i < size_; i++) {
        int old_id = reorder_to_old_[i];
        graph_new_.push_back(Node(static_cast<int>(graph_[old_id].neighbors_.size()) - 1));
        for (int j = 0; j < graph_[old_id].neighbors_.size(); ++j) {
            if (j == 0) {
                graph_new_[i].neighbors_[j].reserve(maxM0_);
            } else {
                graph_new_[i].neighbors_[j].reserve(maxM_);
            }
            for (int k : graph_[old_id].neighbors_[j]) {
                graph_new_[i].neighbors_[j].push_back(reorder_to_new_[k]);
            }
        }
    }
    graph_ = graph_new_;
    enter_point_ = reorder_to_new_[enter_point_];
}

template <typename Space>
inline void HNSW<Space>::ReOrdering() {
    BFSTreeReOrdering();
    // MSTTreeReOrdering();
    GraphReWrite();
}

template <typename Space>
inline IntType HNSW<Space>::DfsStat(IntType cur) {
    dfs_stat_[cur] = 1;
    for (auto next : bfs_tree_[cur]) {
        dfs_stat_[cur] += DfsStat(next);
    }
    return dfs_stat_[cur];
}

template <typename Space>
inline void HNSW<Space>::DfsReorder(IntType cur) {
    std::vector<std::pair<IntType, IntType>> queue;
    for (auto next : bfs_tree_[cur]) {
        queue.push_back({dfs_stat_[next], next});
    }
    std::sort(queue.begin(), queue.end());
    std::reverse(queue.begin(), queue.end());
    for (auto& [prioriter, next] : queue) {
        DfsReorder(next);
    }
    reorder_to_new_[cur] = reorder_num++;
    reorder_to_old_[reorder_to_new_[cur]] = cur;
}

template <typename Space>
inline std::vector<IntType> HNSW<Space>::Search(FloatType* query, IntType K, IntType ef) {
    IntType enter_point = enter_point_;
    for (IntType i = max_level_; i >= 1; --i) {
        enter_point = SearchLayer(query, enter_point, 1, i).top().second;
    }
    auto nearest_neighbours = SearchLayer(query, enter_point, ef, 0);
    while (nearest_neighbours.size() > K) {
        nearest_neighbours.pop();
    }
    std::vector<IntType> array;
    array.reserve(nearest_neighbours.size());
    while (!nearest_neighbours.empty()) {
#ifdef REORDER
        array.push_back(reorder_to_old_[nearest_neighbours.top().second]);
#else
        array.push_back(nearest_neighbours.top().second);
#endif
        nearest_neighbours.pop();
    }
    std::reverse(array.begin(), array.end());
    return array;
}

template <typename Space>
inline void HNSW<Space>::Save(std::ofstream& file) {
    file << size_ << "\n";
    file << enter_point_ << "\n";
    file << M_ << "\n";
    file << ef_construction_ << "\n";
    file << max_level_ << "\n";
    IntType dim = SIZE;
    file << dim << "\n";
    for (IntType node = 0; node < size_; ++node) {
        file << graph_[node].neighbors_.size() << '\n';
        for (IntType level = 0; level < graph_[node].neighbors_.size(); ++level) {
            file << graph_[node].neighbors_[level].size() << " ";
            for (IntType neighbour : graph_[node].neighbors_[level]) {
                file << neighbour << " ";
            }
            file << "\n";
        }
    }
    file << reorder_to_old_.size() << "\n";
    for (IntType node : reorder_to_old_) {
        file << node << " ";
    }
    file << reorder_to_new_.size() << "\n";
    for (IntType node : reorder_to_new_) {
        file << node << " ";
    }
}

template <typename Space>
inline HNSW<Space>::HNSW(std::ifstream& file, std::ifstream& file_data) {
    file >> size_;
    graph_.reserve(size_);
    was_ = std::vector<IntType>(size_);
    max_elements_ = size_;
    file >> enter_point_;
    file >> M_;
    maxM_ = M_;
    maxM0_ = 2 * M_;
    file >> ef_construction_;
    file >> max_level_;
    IntType dim;
    file >> dim;
    for (IntType node = 0; node < size_; ++node) {
        IntType level_number;
        file >> level_number;
        graph_.push_back(Node(level_number - 1));
        for (IntType level = 0; level < level_number; ++level) {
            if (level == 0) {
                graph_[node].neighbors_[level].reserve(maxM0_);
            } else {
                graph_[node].neighbors_[level].reserve(maxM_);
            }
            IntType neighbour_number;
            file >> neighbour_number;
            for (IntType it = 0; it < neighbour_number; ++it) {
                IntType neighbour;
                file >> neighbour;
                graph_[node].neighbors_[level].push_back(neighbour);
            }
        }
    }
    IntType reorder_old_size;
    file >> reorder_old_size;
    reorder_to_old_.resize(reorder_old_size);
    for (int i = 0; i < reorder_old_size; ++i) {
        file >> reorder_to_old_[i];
    }
    IntType reorder_new_size;
    file >> reorder_new_size;
    reorder_to_new_.resize(reorder_new_size);
    for (int i = 0; i < reorder_new_size; ++i) {
        file >> reorder_to_new_[i];
    }
    data_long_ = static_cast<FloatType*>(aligned_alloc(ALIGN64, (size_ * dim) * sizeof(FloatType)));
    for (IntType node = 0; node < size_; ++node) {
        for (IntType i = 0; i < dim; ++i) {
            FloatType x;
            file_data >> x;
            data_long_[reorder_to_new_[node] * SIZE + i] = x;
        }
    }
}
