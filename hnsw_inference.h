#pragma once

#include <string.h>
#include <sys/mman.h>

#include "primitives.h"

void* HugeAlloc(size_t total_size_bytes) {
    size_t hugepage_size = 2 * 1024 * 1024;
    size_t size = (total_size_bytes + hugepage_size - 1) & ~(hugepage_size - 1);
    void* ptr =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    std::cout << ptr << "\n";
    return ptr;
}

template <typename Space>
struct HNSWInference {
    HNSWInference(std::ifstream& file, std::ifstream& file_data);
    ~HNSWInference() {
    }
    void LoadPQ(std::ifstream& file_data_pq, std::ifstream& file_centroids);
    void LoadPQMatrix(std::ifstream& file_matrix);
    void LoadReRankMatrix(std::ifstream& file_matrix);
    // void Transform(FloatType* query);
    void FillTable(FloatType* query);
    FloatType GetDistance(IntType node);
    QueueLess SearchLayer(FloatType* query, IntType enter_point, IntType ef, IntType level);
    QueueLess SearchLayerPQ(FloatType* query, IntType enter_point, IntType ef, IntType level);
    std::vector<IntType> Search(FloatType* query, IntType K, IntType ef);
    std::vector<IntType> SearchPQ(FloatType* query, IntType K, IntType ef);

    IntType M_;
    IntType maxM_;
    IntType maxM0_;
    IntType ef_construction_;
    IntType enter_point_;
    IntType size_ = 0;
    IntType current_was_ = 0;
    IntType max_elements_;
    IntType max_level_ = -1;

    std::vector<IntType> reorder_to_new_;
    std::vector<IntType> reorder_to_old_;

    Space space_;

    FloatType* data;
    IntType* list0;
    IntType** list;
    IntType* was_;

    uint8_t* quantized_data;
    FloatType* centroids;
    std::array<std::array<FloatType, 256>, SUBSPACES> pq_table;
    FloatType* matrix_PQ;
    FloatType* vector_PQ;
    bool need_mm_PQ = false;
    FloatType* matrix_rerank;
    FloatType* vector_rerank;
    bool need_mm_rerank = false;
};

template <typename Space>
inline void HNSWInference<Space>::LoadPQ(std::ifstream& file_data_pq,
                                         std::ifstream& file_centroids) {
    auto ReadBinaryUInt8 = [&file_data_pq](uint8_t& value) {
        file_data_pq.read(reinterpret_cast<char*>(&value), sizeof(uint8_t));
    };

    // quantized_data = (uint8_t*)aligned_alloc(ALIGN64, SUBSPACES * size_ * sizeof(uint8_t));
    quantized_data = (uint8_t*)HugeAlloc(SUBSPACES * size_ * sizeof(uint8_t));

    for (IntType node = 0; node < size_; ++node) {
        uint8_t* pointer = (uint8_t*)(quantized_data + reorder_to_new_[node] * SUBSPACES);
        for (IntType i = 0; i < SUBSPACES; ++i) {
            ReadBinaryUInt8(pointer[i]);
        }
    }

    centroids = (FloatType*)aligned_alloc(ALIGN64, SUBSPACES * BITS * SUBSIZE * sizeof(FloatType));
    file_centroids.read(reinterpret_cast<char*>(centroids),
                        SUBSPACES * BITS * SUBSIZE * sizeof(FloatType));
}

template <typename Space>
inline void HNSWInference<Space>::LoadPQMatrix(std::ifstream& file_matrix) {
    vector_PQ = (FloatType*)aligned_alloc(ALIGN64, SIZE * sizeof(FloatType));
    matrix_PQ = (FloatType*)aligned_alloc(ALIGN64, SIZE * SIZE * sizeof(FloatType));
    file_matrix.read(reinterpret_cast<char*>(matrix_PQ), SIZE * SIZE * sizeof(FloatType));
    need_mm_PQ = true;
}

template <typename Space>
inline void HNSWInference<Space>::LoadReRankMatrix(std::ifstream& file_matrix) {
    vector_rerank = (FloatType*)aligned_alloc(ALIGN64, SIZE * sizeof(FloatType));
    matrix_rerank = (FloatType*)aligned_alloc(ALIGN64, SIZE * SIZE * sizeof(FloatType));
    file_matrix.read(reinterpret_cast<char*>(matrix_rerank), SIZE * SIZE * sizeof(FloatType));
    need_mm_rerank = true;
}

// template <typename Space>
// inline void HNSWInference<Space>::Transform(FloatType* query) {
//     memset(quantized_vector, 0, SIZE * sizeof(FloatType));
//     for (IntType i = 0; i < SIZE; ++i) {
//         for (IntType j = 0; j < SIZE; ++j) {
//             quantized_vector[i] += matrix[i * SIZE + j] * query[j];
//         }
//     }
// }

template <typename Space>
inline void HNSWInference<Space>::FillTable(FloatType* query) {
    if (need_mm_PQ) {
        // Transform(query);
        MatVecMul(matrix_PQ, query, vector_PQ);
        query = vector_PQ;
    }
    FloatType* pointer = centroids;
    for (int i = 0; i < SUBSPACES; ++i) {
        FloatType* subquery = query + i * SUBSIZE;
        for (int code = 0; code < BITS; ++code) {
            pq_table[i][code] = space_.DistanceSubspace(subquery, pointer);
            pointer += SUBSIZE;
        }
    }
}

// template <typename Space>
// inline FloatType HNSWInference<Space>::GetDistance(IntType node) {
//     uint8_t* query = quantized_data + node * SUBSPACES;
//     FloatType result = 0;
//     for (int i = 0; i < SUBSPACES; ++i) {
//         result += pq_table[i][query[i]];
//     }
//     return result;
// }

template <typename Space>
inline FloatType HNSWInference<Space>::GetDistance(IntType node) {
    const uint8_t* query = quantized_data + node * SUBSPACES;
    int i = 0;
    FloatType r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    for (; i + 3 < SUBSPACES; i += 4) {
        r0 += pq_table[i][query[i]];
        r1 += pq_table[i + 1][query[i + 1]];
        r2 += pq_table[i + 2][query[i + 2]];
        r3 += pq_table[i + 3][query[i + 3]];
    }
    FloatType result = r0 + r1 + r2 + r3;
    for (; i < SUBSPACES; ++i) {
        result += pq_table[i][query[i]];
    }
    return result;
}

template <typename Space>
inline QueueLess HNSWInference<Space>::SearchLayer(FloatType* query, IntType enter_point,
                                                   IntType ef, IntType level) {
    was_[enter_point] = ++current_was_;
    QueueGreater candidates;

    FloatType* ep_pointer = data + SIZE * enter_point;
    FloatType enter_point_distance = space_.Distance(ep_pointer, query);

    candidates.emplace(enter_point_distance, enter_point);

    QueueLess nearest_neighbours;
    nearest_neighbours.emplace(enter_point_distance, enter_point);

    while (!candidates.empty()) {
        IntType current = candidates.top().second;
        FloatType current_distance = candidates.top().first;
        candidates.pop();
        FloatType furthest_distance = nearest_neighbours.top().first;

        if (current_distance > furthest_distance) {
            break;
        }

        IntType* pointer;
        if (level == 0) {
            pointer = list0 + (maxM0_ + 1) * current;
        } else {
            pointer = list[current] + (level - 1) * (maxM_ + 1);
        }

        IntType cursz = pointer[0];

        _mm_prefetch((char*)(was_ + *(pointer + 1)), _MM_HINT_T0);
        _mm_prefetch((char*)(data + (*(pointer + 1)) * SIZE), _MM_HINT_T0);

        for (IntType i = 1; i <= cursz; ++i) {
            IntType next = pointer[i];

            _mm_prefetch((char*)(was_ + *(pointer + i + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(data + (*(pointer + i + 1)) * SIZE), _MM_HINT_T0);

            if (was_[next] != current_was_) {
                was_[next] = current_was_;
                furthest_distance = nearest_neighbours.top().first;

                FloatType* ne_pointer = data + SIZE * next;
                FloatType distance = space_.Distance(ne_pointer, query);

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
inline QueueLess HNSWInference<Space>::SearchLayerPQ(FloatType* query, IntType enter_point,
                                                     IntType ef, IntType level) {
    was_[enter_point] = ++current_was_;
    QueueGreater candidates;
    FloatType enter_point_distance = GetDistance(enter_point);

    candidates.emplace(enter_point_distance, enter_point);

    QueueLess nearest_neighbours;
    nearest_neighbours.emplace(enter_point_distance, enter_point);

    while (!candidates.empty()) {
        IntType current = candidates.top().second;
        FloatType current_distance = candidates.top().first;
        candidates.pop();
        FloatType furthest_distance = nearest_neighbours.top().first;

        if (current_distance > furthest_distance) {
            break;
        }

        IntType* pointer;
        if (level == 0) {
            pointer = list0 + (maxM0_ + 1) * current;
        } else {
            pointer = list[current] + (level - 1) * (maxM_ + 1);
        }

        IntType cursz = pointer[0];

        _mm_prefetch((char*)(was_ + *(pointer + 1)), _MM_HINT_T0);
        _mm_prefetch((char*)(quantized_data + (*(pointer + 1)) * SUBSPACES), _MM_HINT_T0);

        for (IntType i = 1; i <= cursz; ++i) {
            IntType next = pointer[i];

            _mm_prefetch((char*)(was_ + *(pointer + i + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(quantized_data + (*(pointer + i + 1)) * SUBSPACES), _MM_HINT_T0);

            if (was_[next] != current_was_) {
                was_[next] = current_was_;
                furthest_distance = nearest_neighbours.top().first;

                FloatType distance = GetDistance(next);

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
inline std::vector<IntType> HNSWInference<Space>::Search(FloatType* query, IntType K, IntType ef) {
    IntType enter_point = enter_point_;
    for (IntType i = max_level_; i >= 1; --i) {
        enter_point = SearchLayer(query, enter_point, 1, i).top().second;
    }
    // int dst_base = space_.Distance(data + SIZE * enter_point, query);
    // for (int it : grid) {
    //     int dst_cur = space_.Distance(data + SIZE * it, query);
    //     if (dst_cur < dst_base) {
    //         dst_base = dst_cur;
    //         enter_point = it;
    //     }
    // }
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
inline std::vector<IntType> HNSWInference<Space>::SearchPQ(FloatType* query, IntType K,
                                                           IntType ef) {
    FillTable(query);
    IntType enter_point = enter_point_;
    for (IntType i = max_level_; i >= 1; --i) {
        enter_point = SearchLayerPQ(query, enter_point, 1, i).top().second;
    }
    auto nearest_neighbours = SearchLayerPQ(query, enter_point, ef, 0);
    if (need_mm_rerank) {
        MatVecMul(matrix_rerank, query, vector_rerank);
        query = vector_rerank;
    }
    std::vector<std::pair<FloatType, IntType>> candidates;
    candidates.reserve(nearest_neighbours.size());
    while (!nearest_neighbours.empty()) {
        IntType next = nearest_neighbours.top().second;
        FloatType* ne_pointer = data + SIZE * next;
        FloatType distance = space_.Distance(ne_pointer, query);
        nearest_neighbours.pop();
        candidates.push_back({distance, next});
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<IntType> array;
    array.reserve(nearest_neighbours.size());
    for (int i = 0; i < std::min(K, (IntType)candidates.size()); ++i) {
        array.push_back(reorder_to_old_[candidates[i].second]);
    }
    return array;
}

template <typename Space>
inline HNSWInference<Space>::HNSWInference(std::ifstream& file, std::ifstream& file_data) {
    auto ReadBinaryInt = [&file](IntType& value) {
        file.read(reinterpret_cast<char*>(&value), sizeof(IntType));
    };

    ReadBinaryInt(size_);
    max_elements_ = size_;
    ReadBinaryInt(enter_point_);
    ReadBinaryInt(M_);
    maxM_ = M_;
    maxM0_ = 2 * M_;
    ReadBinaryInt(ef_construction_);
    ReadBinaryInt(max_level_);
    IntType dim;
    ReadBinaryInt(dim);
    was_ = (IntType*)aligned_alloc(ALIGN64, size_ * sizeof(IntType));
    memset(was_, 0, size_ * sizeof(int));

    data = (FloatType*)HugeAlloc(size_ * SIZE * sizeof(FloatType));

    list = (IntType**)(aligned_alloc(ALIGN64, size_ * sizeof(IntType*)));
    list0 = (IntType*)(aligned_alloc(ALIGN64, size_ * (maxM0_ + 1) * sizeof(IntType)));

    IntType degree_sum = 0;
    for (IntType node = 0; node < size_; ++node) {
        IntType level_number;
        ReadBinaryInt(level_number);
        if (level_number > 1) {
            list[node] = (IntType*)(aligned_alloc(
                ALIGN4, (level_number - 1) * (maxM_ + 1) * sizeof(IntType)));
        }
        for (IntType level = 0; level < level_number; ++level) {
            IntType neighbour_number;
            ReadBinaryInt(neighbour_number);
            if (level == 0) {
                degree_sum += neighbour_number;
            }
            IntType* pointer;
            if (level == 0) {
                pointer = (IntType*)(list0 + (maxM0_ + 1) * node);
            } else {
                pointer = (IntType*)(list[node] + (level - 1) * (maxM_ + 1));
            }
            pointer[0] = neighbour_number;

            for (IntType it = 0; it < neighbour_number; ++it) {
                IntType neighbour;
                ReadBinaryInt(neighbour);
                pointer[it + 1] = neighbour;
            }
        }
    }
    std::cout << "avg degree " << static_cast<FloatType>(degree_sum) / size_ << "\n";
    std::cout << "maxM: " << maxM_ << ", maxM0: " << maxM0_ << "\n";

    IntType reorder_old_size;
    ReadBinaryInt(reorder_old_size);
    reorder_to_old_.resize(reorder_old_size);
    for (IntType i = 0; i < reorder_old_size; ++i) {
        ReadBinaryInt(reorder_to_old_[i]);
    }

    IntType reorder_new_size;
    ReadBinaryInt(reorder_new_size);
    reorder_to_new_.resize(reorder_new_size);
    for (IntType i = 0; i < reorder_new_size; ++i) {
        ReadBinaryInt(reorder_to_new_[i]);
    }

    auto ReadBinaryFloat = [&file_data](FloatType& value) {
        file_data.read(reinterpret_cast<char*>(&value), sizeof(FloatType));
    };

    for (IntType node = 0; node < size_; ++node) {
        FloatType* pointer = (FloatType*)(data + SIZE * reorder_to_new_[node]);
        for (IntType i = 0; i < dim; ++i) {
            ReadBinaryFloat(pointer[i]);
        }
    }
}
