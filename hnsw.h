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

#define REORDER

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
        data_long_ = static_cast<float*>(aligned_alloc(64, (max_elements * SIZE) * sizeof(float)));
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
    void TreeReOrdering();
    void SumOfModulesReOrdering();
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
    float ssg_cos = 0.5;
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
        for(int i=0; i<cursz; ++i){
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
    // bool simple = false;
    // if (simple) {
    //     std::vector<IntType> array;
    //     array.reserve(maxM + 1);
    //     while (!candidates.empty()) {
    //         array.push_back(candidates.top().second);
    //         candidates.pop();
    //     }
    //     std::reverse(array.begin(), array.end());
    //     array.resize(std::min(M, static_cast<IntType>(array.size())));
    //     return array;
    // }
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

// template <typename Space>
// inline std::vector<IntType> HNSW<Space>::SSG(IntType node, std::vector<IntType>& candidates,
//                                              IntType M) {

//     std::sort(candidates.begin(), candidates.end());
//     int new_size = std::unique(candidates.begin(), candidates.end()) - candidates.begin();
//     candidates.resize(new_size);

//     QueueLess q;
//     for (IntType neighbour : candidates) {
//         q.emplace(space_.Distance(data_[node], data_[neighbour]), neighbour);
//     }

//     std::vector<std::pair<FloatType, IntType>> queue;
//     queue.reserve(q.size());
//     while (!q.empty()) {
//         queue.push_back(q.top());
//         q.pop();
//     }
//     std::reverse(queue.begin(), queue.end());
//     std::vector<IntType> selected;
//     std::vector<Point> dir;
//     dir.reserve(M);
//     selected.reserve(M + 1);
//     for (auto& [distance, element] : queue) {
//         if (selected.size() >= M) {
//             break;
//         }
//         if (element == node)
//             continue;
//         Point curdir = data_[element] - data_[node];
//         curdir.Normalize();
//         bool good = true;
//         for (IntType i = 0; i < selected.size(); ++i) {
//             IntType neighbour = selected[i];
//             FloatType cos = space_.Cos(dir[i], curdir);
//             if (cos > ssg_cos) {
//                 good = false;
//                 break;
//             }
//         }
//         if (good) {
//             selected.push_back(element);
//             dir.push_back(curdir);
//         }
//     }
//     return selected;
// }

template <typename Space>
inline void HNSW<Space>::TreeReOrdering() {
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
inline void HNSW<Space>::SumOfModulesReOrdering() {
    std::vector<std::vector<IntType>> graph_inv_(size_);
    for (int i = 0; i < size_; ++i) {
        for (int j : graph_[i].neighbors_[0]) {
            graph_inv_[j].push_back(i);
        }
    }
    auto ScoreF = [=](int x, int y) { return abs(reorder_to_new_[x] - reorder_to_new_[y]); };

    auto GetScore = [=](int i, int j) {
        int64_t score = 0;
        for (int k : graph_[i].neighbors_[0]) {
            score += ScoreF(i, k);
        }
        for (int k : graph_[j].neighbors_[0]) {
            score += ScoreF(j, k);
        }
        for (int k : graph_inv_[i]) {
            score += ScoreF(i, k);
        }
        for (int k : graph_inv_[j]) {
            score += ScoreF(j, k);
        }
        return score;
    };

    int64_t sum = 0;
    for (int i = 0; i < size_; ++i) {
        for (int j : graph_[i].neighbors_[0]) {
            sum += ScoreF(i, j);
        }
    }
    std::cout << sum << "\n";
    std::cout << "START\n";
    for (int _ = 0; _ < 100; _++) {
        std::cout << _ << "\n";
        for (int i = 0; i < size_; ++i) {
            if (i % 10000 == 0) {
                std::cout << i << "\n";
            }
            for (int j = reorder_to_new_[i] + 1; (j <= reorder_to_new_[i] + 30) and (j < size_);
                 ++j) {
                int64_t l = reorder_to_old_[j];
                int64_t mn_score = 0, index = -1;
                for (int ne : graph_[i].neighbors_[0]) {
                    int64_t score_old = GetScore(l, ne);
                    std::swap(reorder_to_new_[l], reorder_to_new_[ne]);
                    int64_t score_new = GetScore(l, ne);
                    if (score_new - score_old < mn_score) {
                        mn_score = score_new - score_old;
                        index = ne;
                    }
                    std::swap(reorder_to_new_[l], reorder_to_new_[ne]);
                }
                if (mn_score < 0) {
                    std::swap(reorder_to_new_[l], reorder_to_new_[index]);
                    std::swap(reorder_to_old_[reorder_to_new_[l]],
                              reorder_to_old_[reorder_to_new_[index]]);
                }
            }
        }
        sum = 0;
        for (int i = 0; i < size_; ++i) {
            for (int j : graph_[i].neighbors_[0]) {
                sum += ScoreF(i, j);
            }
        }
        std::cout << sum << "\n";
    }
}

template <typename Space>
inline void HNSW<Space>::ReOrdering() {
    SumOfModulesReOrdering();
    // TreeReOrdering();
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

// template <typename Space>
// inline void HNSW<Space>::Improve() {
//     int l = 100, r = 50;
//     std::vector<std::vector<IntType>> ne(size_);
//     for (IntType node = 0; node < size_; ++node) {
//         if (node % 10000 == 0) {
//             std::cout << node << "\n";
//         }
//         for (IntType level = 0; level < 1; ++level) {
//             std::vector<IntType> candidates;
//             candidates.reserve(l);
//             for (IntType x : graph_[node].neighbors_[level]) {
//                 candidates.push_back(x);
//                 for (IntType y : graph_[x].neighbors_[level]) {
//                     candidates.push_back(y);
//                 }
//             }
//             ne[node] = SSG(node, candidates, r);
//         }
//     }
//     for (IntType node = 0; node < size_; ++node) {
//         graph_[node].neighbors_[0] = ne[node];
//     }
//     for (IntType node = 0; node < size_; ++node) {
//         if (node % 10000 == 0) {
//             std::cout << node << "\n";
//         }
//         for (IntType level = 0; level < 1; ++level) {
//             for (IntType x : graph_[node].neighbors_[level]) {
//                 graph_[x].neighbors_[level].push_back(node);
//                 graph_[x].neighbors_[level] = SSG(x, graph_[x].neighbors_[level], r);
//             }
//         }
//     }
// }

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
    data_long_ = static_cast<float*>(aligned_alloc(64, (size_ * dim) * sizeof(float)));
    for (IntType node = 0; node < size_; ++node) {
        for (IntType i = 0; i < dim; ++i) {
            FloatType x;
            file_data >> x;
            data_long_[reorder_to_new_[node] * SIZE + i] = x;
        }
    }
}
