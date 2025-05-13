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

// #define MEMORY_OPTIMIZATION
// #define LONG_VECTOR
#define REORDER
// #define THRESHOLD 100000

typedef std::priority_queue<std::pair<FloatType, IntType>,
                            std::vector<std::pair<FloatType, IntType>>,
                            std::less<std::pair<FloatType, IntType>>>
    QueueLess;

typedef std::priority_queue<std::pair<FloatType, IntType>,
                            std::vector<std::pair<FloatType, IntType>>,
                            std::greater<std::pair<FloatType, IntType>>>
    QueueGreater;

struct Node {
    Node(IntType max_level) {
        neighbors_.resize(max_level + 1);
    }
    std::vector<std::vector<IntType>> neighbors_;
};

template <typename Space>
struct HNSW {
    HNSW(IntType M, IntType ef_construction, IntType max_elements)
        : M_{M},
          maxM_{M},
          maxM0_{2 * M},
          ef_construction_{ef_construction},
          max_elements_{max_elements} {
        was_ = std::vector<IntType>(max_elements_);
        // data_.reserve(max_elements_);
        graph_.reserve(max_elements_);
        data_long_ = static_cast<float*>(aligned_alloc(64, (max_elements * SIZE) * sizeof(float)));
    }
    HNSW(std::ifstream& file);
    ~HNSW() {
        free(data_long_);
    }
    void Add(int level);
    QueueLess SearchLayer(FloatType* query, IntType enter_point, IntType ef, IntType level);
    // QueueLess SearchLayer0(Point& query, IntType enter_point, IntType ef, IntType level);
    std::vector<IntType> SelectNeighbours(QueueLess& candidates, IntType M, IntType maxM);
    std::vector<IntType> Search(FloatType* query, IntType K, IntType ef);
    std::vector<IntType> SSG(IntType node, std::vector<IntType>& candidates, IntType M);
    // void MemoryManager(IntType upper_threshold = 200000);
    // void AddNeighborhood(IntType i, IntType j);
    // void AddNeighborhood(IntType i);
    void TreeReOrdering();
    void SumOfModulesReOrdering();
    // void SumOfAbs();
    void GraphReWrite();
    void ReOrdering();
    IntType DfsStat(IntType cur);
    void DfsReorder(IntType cur);

    void Improve();
    void Save(std::ofstream& file, IntType precision = 1);
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

    // std::vector<IntType> A0_;
    // std::vector<IntType> B0_;

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
    // data_.push_back(point);
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
        for (IntType next : graph_[current].neighbors_[level]) {
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

// template <typename Space>
// inline QueueLess HNSW<Space>::SearchLayer0(Point& query, IntType enter_point, IntType ef,
//                                            IntType level) {
//     was_[enter_point] = ++current_was_;
//     QueueGreater candidates;
//     FloatType enter_point_distance = space_.Distance(data_[enter_point], query);
//     candidates.emplace(enter_point_distance, enter_point);
//     QueueLess nearest_neighbours;
//     nearest_neighbours.emplace(enter_point_distance, enter_point);
//     while (!candidates.empty()) {
//         IntType current = candidates.top().second;
//         candidates.pop();
//         FloatType furthest_distance = nearest_neighbours.top().first;
//         if (space_.Distance(data_[current], query) > furthest_distance and
//             nearest_neighbours.size() == ef) {
//             break;
//         }
// #ifdef LONG_VECTOR
//         for (IntType i = current * maxM0_; i < current * maxM0_ + B0_[current]; ++i)
// #else
//         for (IntType i = B0_[2 * current]; i < B0_[2 * current + 1]; ++i)
// #endif
//         {
//             IntType next = A0_[i];
//             if (was_[next] != current_was_) {
//                 was_[next] = current_was_;
//                 furthest_distance = nearest_neighbours.top().first;
//                 FloatType distance = space_.Distance(data_[next], query);
//                 if (distance < furthest_distance or nearest_neighbours.size() < ef) {
//                     candidates.emplace(distance, next);
//                     nearest_neighbours.emplace(distance, next);
//                     if (nearest_neighbours.size() > ef) {
//                         nearest_neighbours.pop();
//                     }
//                 }
//             }
//         }
//     }
//     return nearest_neighbours;
// }

template <typename Space>
inline std::vector<IntType> HNSW<Space>::SelectNeighbours(QueueLess& candidates, IntType M,
                                                          IntType maxM) {
    bool simple = true;
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

// template <typename Space>
// inline void HNSW<Space>::MemoryManager(IntType upper_threshold) {
// #ifdef LONG_VECTOR
//     A0_ = std::vector<IntType>(maxM0_ * size_);
//     B0_ = std::vector<IntType>(size_);
//     for (IntType i = 0; i < size_; ++i) {
//         for (int j = 0; j < graph_[i].neighbors_[0].size(); ++j) {
//             A0_[i * maxM0_ + j] = graph_[i].neighbors_[0][j];
//         }
//         B0_[i] = graph_[i].neighbors_[0].size();
//     }
// #else
//     std::priority_queue<std::pair<IntType, std::pair<IntType, IntType>>,
//                         std::vector<std::pair<IntType, std::pair<IntType, IntType>>>,
//                         std::less<std::pair<IntType, std::pair<IntType, IntType>>>>
//         q;
//     A0_.resize(0);
//     B0_ = std::vector<IntType>(2 * size_);
//     for (IntType i = 0; i < size_; ++i) {
//         if (i % 10000 == 0) {
//             std::cout << i << "\n";
//         }
//         std::set<int> candidates_;
//         for (auto x : graph_[i].neighbors_[0]) {
//             if (i != x) {
//                 candidates_.insert(x);
//             }
//             // for (auto y : graph_[x].neighbors_[0])
//             // {
//             //     if (i != y)
//             //     {
//             //         candidates_.insert(y);
//             //     }
//             // }
//         }

//         std::set<int> cur;
//         for (auto x : graph_[i].neighbors_[0]) {
//             cur.insert(x);
//         }
//         std::set<std::pair<IntType, std::pair<IntType, IntType>>> nq;
//         for (auto x : candidates_) {
//             int count = 0;
//             for (auto y : graph_[x].neighbors_[0]) {
//                 count += cur.count(y);
//             }
//             nq.insert({count, {i, x}});
//             if (nq.size() > 5) {
//                 nq.erase(nq.begin());
//             }
//         }
//         for (auto x : nq) {
//             q.push(x);
//         }
//     }
//     int cnt = 0;
//     int num = 0;
//     std::vector<int> was(size_, -1);
//     while (!q.empty()) {
//         if (num >= upper_threshold) {
//             break;
//         }
//         auto t = q.top();
//         q.pop();
//         auto [i, j] = t.second;
//         if (was[i] == -1 and was[j] == -1) {
//             if (i > j) {
//                 std::swap(i, j);
//             }
//             was[i] = j;
//             was[j] = j;
//             cnt += t.first;
//             num++;
//         }
//     }
//     std::cout << cnt << "\n";
//     int res = 0;
//     for (int i = 0; i < size_; ++i) {
//         if (was[i] == -1) {
//             AddNeighborhood(i);
//             res++;
//         } else if (was[i] != i) {
//             AddNeighborhood(i, was[i]);
//         }
//     }
//     std::cout << res << "\n";

//     A0_.shrink_to_fit();
// #endif
//     for (int i = 0; i < size_; ++i) {
//         graph_[i].neighbors_[0].clear();
//         graph_[i].neighbors_[0].shrink_to_fit();
//     }
// }

// template <typename Space>
// inline void HNSW<Space>::AddNeighborhood(IntType i, IntType j) {
//     std::set<IntType> i_minus_j;
//     std::set<IntType> j_minus_i;
//     std::set<IntType> intersection;
//     for (IntType element : graph_[i].neighbors_[0]) {
//         i_minus_j.insert(element);
//     }
//     for (IntType element : graph_[j].neighbors_[0]) {
//         if (i_minus_j.count(element)) {
//             intersection.insert(element);
//         } else {
//             j_minus_i.insert(element);
//         }
//     }
//     for (IntType element : intersection) {
//         i_minus_j.erase(element);
//     }
//     B0_[2 * i] = size0_;
//     for (auto x : i_minus_j) {
//         A0_.push_back(x);
//         size0_++;
//     }
//     B0_[2 * j] = size0_;
//     for (auto x : intersection) {
//         A0_.push_back(x);
//         size0_++;
//     }
//     B0_[2 * i + 1] = size0_;
//     for (auto x : j_minus_i) {
//         A0_.push_back(x);
//         size0_++;
//     }
//     B0_[2 * j + 1] = size0_;
// }

// template <typename Space>
// inline void HNSW<Space>::AddNeighborhood(IntType i) {
//     B0_[2 * i] = size0_;
//     for (auto x : graph_[i].neighbors_[0]) {
//         A0_.push_back(x);
//         size0_++;
//     }
//     B0_[2 * i + 1] = size0_;
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
    // Point meansOne(means1.size());
    // for (int i = 0; i < means1.size(); ++i) {
    //     meansOne[i] = means1[i];
    // }
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
    // std::vector<Point> data_new_(size_, Point(data_[0].Size()));
    std::vector<FloatType> data_new(SIZE * size_);
    for (int i = 0; i < size_; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            data_new[i * SIZE + j] = data_long_[reorder_to_old_[i] * SIZE + j];
        }
    }
    for (int i = 0; i < SIZE * size_; ++i) {
        data_long_[i] = data_new[i];
    }
    // data_ = data_new_;
}

#define SUM_ABS

template <typename Space>
inline void HNSW<Space>::SumOfModulesReOrdering() {
    reorder_to_new_ = std::vector<IntType>(size_);
    for (int i = 0; i < size_; ++i) {
        reorder_to_new_[i] = i;
    }
    reorder_to_old_ = std::vector<IntType>(size_);
    for (int i = 0; i < size_; ++i) {
        reorder_to_old_[i] = i;
    }

    std::vector<std::vector<IntType>> graph_inv_(size_);
    for (int i = 0; i < size_; ++i) {
        for (int j : graph_[i].neighbors_[0]) {
            graph_inv_[j].push_back(i);
        }
    }
    // int l_score = 10;
    // int k_score = size;
    // std::cout << k_score << "\n";
#ifdef SUM_ABS
    // sum of abs
    auto ScoreF = [=](int x, int y) {
        // return ((reorder_to_new_[x]/k_score) != (reorder_to_new_[y]/k_score));
        return abs(reorder_to_new_[x] - reorder_to_new_[y]);
        // return (abs(reorder_to_new_[x] - reorder_to_new_[y]) > THRESHOLD) *
        //        abs(reorder_to_new_[x] - reorder_to_new_[y]);
    };

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
#else
    // sum of max
    auto ScoreF = [=](int x) {
        int mx = 0;
        for (int k : graph_[x].neighbors_[0]) {
            mx = std::max(mx, abs(reorder_to_new_[x] - reorder_to_new_[k]));
        }
        return mx;
    };

    auto GetScore = [=](int i, int j) {
        int64_t score = ScoreF(i) + ScoreF(j);
        for (int k : graph_[i].neighbors_[0]) {
            score += ScoreF(k);
        }
        for (int k : graph_[j].neighbors_[0]) {
            score += ScoreF(k);
        }
        for (int k : graph_inv_[i]) {
            score += ScoreF(k);
        }
        for (int k : graph_inv_[j]) {
            score += ScoreF(k);
        }
        return score;
    };
#endif

    int64_t sum = 0;
    for (int i = 0; i < size_; ++i) {
        for (int j : graph_[i].neighbors_[0]) {
#ifdef SUM_ABS
            sum += ScoreF(i, j);
#else
            sum += ScoreF(i);
#endif
        }
    }
    std::cout << sum << "\n";
    std::cout << "START\n";
    std::mt19937 gen(0);
    std::uniform_int_distribution<long long> dist(0, size_ - 1);
    for (int _ = 0; _ < 50; _++) {
        std::cout << _ << "\n";
        for (int i = 0; i < size_; ++i) {
            if (i % 10000 == 0) {
                std::cout << i << "\n";
            }
            // for (int j = reorder_to_new_[i] + 1; (j <= reorder_to_new_[i] + 30) and (j < size_);
            //      ++j) {
            for (int _ = 0; _ < 30; ++_) {
                // for (int jj = 2; jj <= 61; ++jj) {
                //     int sign = 1;
                //     if (jj % 2)
                //         sign = -1;
                //     int j = reorder_to_new_[i] + sign * (jj / 2);
                //     if (!(0 <= j and j < size_))
                //         continue;
                // if (i == j) {
                //     continue;
                // }
                int j = dist(gen);
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
        for (int ii = 0; ii < size_; ++ii) {
            for (int jj : graph_[ii].neighbors_[0]) {
#ifdef SUM_ABS
                sum += ScoreF(ii, jj);
#else
                sum += ScoreF(ii);
#endif
            }
        }
        std::cout << sum << "\n";
    }
    for (int _ = 0; _ < 20; _++) {
        std::cout << _ << "\n";
        for (int i = 0; i < size_; ++i) {
            if (i % 10000 == 0) {
                std::cout << i << "\n";
            }
            for (int j = reorder_to_new_[i] + 1; (j <= reorder_to_new_[i] + 30) and (j < size_);
                 ++j) {
                // for (int _=0; _<30; ++_){
                // for (int jj = 2; jj <= 61; ++jj) {
                //     int sign = 1;
                //     if (jj % 2)
                //         sign = -1;
                //     int j = reorder_to_new_[i] + sign * (jj / 2);
                //     if (!(0 <= j and j < size_))
                //         continue;
                // if (i == j) {
                //     continue;
                // }
                // int j=dist(gen);
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
        for (int ii = 0; ii < size_; ++ii) {
            for (int jj : graph_[ii].neighbors_[0]) {
#ifdef SUM_ABS
                sum += ScoreF(ii, jj);
#else
                sum += ScoreF(ii);
#endif
            }
        }
        std::cout << sum << "\n";
    }
}

template <typename Space>
inline void HNSW<Space>::ReOrdering() {
    // SumOfModulesReOrdering();
    TreeReOrdering();
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
#ifdef MEMORY_OPTIMIZATION
    auto nearest_neighbours = SearchLayer0(query, enter_point, ef, 0);
#else
    auto nearest_neighbours = SearchLayer(query, enter_point, ef, 0);
#endif
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
inline void HNSW<Space>::Improve() {
    int l = 100, r = 50;
    std::vector<std::vector<IntType>> ne(size_);
    for (IntType node = 0; node < size_; ++node) {
        if (node % 10000 == 0) {
            std::cout << node << "\n";
        }
        for (IntType level = 0; level < 1; ++level) {
            std::vector<IntType> candidates;
            candidates.reserve(l);
            for (IntType x : graph_[node].neighbors_[level]) {
                candidates.push_back(x);
                for (IntType y : graph_[x].neighbors_[level]) {
                    candidates.push_back(y);
                }
            }
            ne[node] = SSG(node, candidates, r);
        }
    }
    for (IntType node = 0; node < size_; ++node) {
        graph_[node].neighbors_[0] = ne[node];
    }
    for (IntType node = 0; node < size_; ++node) {
        if (node % 10000 == 0) {
            std::cout << node << "\n";
        }
        for (IntType level = 0; level < 1; ++level) {
            for (IntType x : graph_[node].neighbors_[level]) {
                graph_[x].neighbors_[level].push_back(node);
                graph_[x].neighbors_[level] = SSG(x, graph_[x].neighbors_[level], r);
            }
        }
    }
}

template <typename Space>
inline void HNSW<Space>::Save(std::ofstream& file, IntType precision) {
    file << size_ << "\n";
    file << enter_point_ << "\n";
    file << M_ << "\n";
    file << ef_construction_ << "\n";
    file << max_level_ << "\n";
    IntType dim = SIZE;
    file << dim << "\n";
    file << std::fixed << std::setprecision(precision);
    for (IntType node = 0; node < size_; ++node) {
        for (IntType i = 0; i < dim; ++i) {
            file << data_long_[node * SIZE + i] << " ";
        }
        file << "\n";
    }
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
    // file << A0_.size() << "\n";
    // for (IntType node : A0_) {
    //     file << node << " ";
    // }
    // file << "\n";
    // file << B0_.size() << "\n";
    // for (IntType node : B0_) {
    //     file << node << " ";
    // }
    // file << "\n";
}

template <typename Space>
inline HNSW<Space>::HNSW(std::ifstream& file) {
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
    // data_ = std::vector<Point>(size_, Point(dim));
    // data_long_ = std::vector<FloatType>(size_ * SIZE);
    data_long_ = static_cast<float*>(aligned_alloc(64, (size_ * SIZE) * sizeof(float)));
    for (IntType node = 0; node < size_; ++node) {
        for (IntType i = 0; i < dim; ++i) {
            FloatType x;
            file >> x;
            data_long_[node * SIZE + i] = x;
            // data_[node][i] = x;
        }
    }
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
            // std::sort(graph_[node].neighbors_[level].begin(),
            // graph_[node].neighbors_[level].end());
        }
    }
    IntType reorder_old_size;
    file >> reorder_old_size;
    reorder_to_old_.resize(reorder_old_size);
    for (int i = 0; i < reorder_old_size; ++i) {
        file >> reorder_to_old_[i];
    }
    // IntType A0size;
    // file >> A0size;
    // A0_.resize(A0size);
    // for (int i = 0; i < A0size; ++i) {
    //     file >> A0_[i];
    // }
    // IntType B0size;
    // file >> B0size;
    // B0_.resize(B0size);
    // for (int i = 0; i < B0size; ++i) {
    //     file >> B0_[i];
    // }
}
