#pragma once

#include <string.h>

#include "primitives.h"

#define REORDER
#define ALIGN64 64
#define ALIGN4 4

template <typename Space>
struct HNSWInference {
    HNSWInference(std::ifstream& file, std::ifstream& file_data);
    ~HNSWInference() {
    }
    QueueLess SearchLayer(FloatType* query, IntType enter_point, IntType ef, IntType level);
    std::vector<IntType> Search(FloatType* query, IntType K, IntType ef);

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

    IntType struct_size;
    IntType struct_shift;

    IntType list_size;

    char* data;
    char** list;
    IntType* was_;
};

template <typename Space>
inline QueueLess HNSWInference<Space>::SearchLayer(FloatType* query, IntType enter_point,
                                                   IntType ef, IntType level) {
    was_[enter_point] = ++current_was_;
    QueueGreater candidates;

    FloatType* ep_pointer = (FloatType*)(data + struct_size * enter_point + struct_shift);
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
            pointer = (IntType*)(data + struct_size * current);
        } else {
            pointer = (IntType*)(list[current] + (level - 1) * list_size);
        }

        IntType cursz = pointer[0];

        _mm_prefetch((char*)(was_ + *(pointer + 1)), _MM_HINT_T0);
        _mm_prefetch(data + (*(pointer + 1)) * struct_size + struct_shift, _MM_HINT_T0);

        for (IntType i = 1; i <= cursz; ++i) {
            IntType next = pointer[i];

            _mm_prefetch((char*)(was_ + *(pointer + i + 1)), _MM_HINT_T0);
            _mm_prefetch(data + (*(pointer + i + 1)) * struct_size + struct_shift, _MM_HINT_T0);

            if (was_[next] != current_was_) {
                was_[next] = current_was_;
                furthest_distance = nearest_neighbours.top().first;

                FloatType* ne_pointer = (FloatType*)(data + struct_size * next + struct_shift);
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
inline std::vector<IntType> HNSWInference<Space>::Search(FloatType* query, IntType K, IntType ef) {
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
inline HNSWInference<Space>::HNSWInference(std::ifstream& file, std::ifstream& file_data) {
    file >> size_;
    max_elements_ = size_;

    file >> enter_point_;
    file >> M_;
    maxM_ = M_;
    maxM0_ = 2 * M_;

    file >> ef_construction_;
    file >> max_level_;

    IntType dim;
    file >> dim;

    struct_size = (maxM0_ + 1) * sizeof(IntType) + SIZE * sizeof(FloatType);
    struct_shift = (maxM0_ + 1) * sizeof(IntType);
    list_size = (maxM_ + 1) * sizeof(IntType);

    was_ = (IntType*)aligned_alloc(ALIGN64, size_ * sizeof(IntType));
    memset(was_, 0, size_ * sizeof(int));

    data = (char*)(aligned_alloc(ALIGN64, struct_size * size_));
    list = (char**)(aligned_alloc(ALIGN64, size_ * sizeof(char*)));

    for (IntType node = 0; node < size_; ++node) {
        IntType level_number;
        file >> level_number;
        if (level_number > 1) {
            list[node] = static_cast<char*>(aligned_alloc(ALIGN4, (level_number - 1) * list_size));
        }
        for (IntType level = 0; level < level_number; ++level) {
            IntType neighbour_number;
            file >> neighbour_number;

            IntType* pointer;
            if (level == 0) {
                pointer = (IntType*)(data + struct_size * node);
            } else {
                pointer = (IntType*)(list[node] + (level - 1) * list_size);
            }
            pointer[0] = neighbour_number;

            for (IntType it = 0; it < neighbour_number; ++it) {
                IntType neighbour;
                file >> neighbour;
                pointer[it + 1] = neighbour;
            }
        }
    }

    IntType reorder_old_size;
    file >> reorder_old_size;
    reorder_to_old_.resize(reorder_old_size);
    for (IntType i = 0; i < reorder_old_size; ++i) {
        file >> reorder_to_old_[i];
    }

    IntType reorder_new_size;
    file >> reorder_new_size;
    reorder_to_new_.resize(reorder_new_size);
    for (IntType i = 0; i < reorder_new_size; ++i) {
        file >> reorder_to_new_[i];
    }

    for (IntType node = 0; node < size_; ++node) {
        FloatType* pointer =
            (FloatType*)(data + struct_size * reorder_to_new_[node] + struct_shift);
        for (IntType i = 0; i < dim; ++i) {
            FloatType x;
            file_data >> x;
            pointer[i] = x;
        }
    }
}
