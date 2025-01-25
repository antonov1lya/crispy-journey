#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <iostream>

#include "primitives.h"

struct Layer
{
    std::unordered_map<size_t, std::vector<size_t>> graph_;
};

template <typename Space>
struct HNSW
{
    HNSW(size_t M, size_t ef_construction, size_t max_elements) : M_{M}, maxM_{M}, maxM0_{2 * M},
                                             ef_construction_{ef_construction}, max_elements_{max_elements}
    {
        was_ = std::vector<size_t>(max_elements_);
    }
    void Add(const Point &point, int level);
    std::vector<size_t> SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level);
    std::vector<size_t> SelectNeighboursSimple(size_t query, std::vector<size_t> candidates, size_t M);
    std::vector<size_t> Search(size_t query, size_t K, size_t ef);
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;
    std::vector<size_t> enter_points_;
    size_t size_ = 0;
    int max_level_ = -1;
    std::vector<Point> data_;
    std::vector<Layer> layers_;
    Space space_;
    std::vector<size_t>was_;
    size_t current_was_;
    size_t max_elements_;
};

template <typename Space>
void HNSW<Space>::Add(const Point &point, int level)
{
    data_.push_back(point);
    size_t index = size_;
    ++size_;
    if (level > max_level_)
    {
        for (int i = max_level_ + 1; i <= level; ++i)
        {
            layers_.push_back(Layer());
            enter_points_.push_back(index);
        }
        max_level_ = level;
    }
    size_t enter_point = enter_points_[max_level_];
    for (int i = max_level_; i > level; --i)
    {
        enter_point = SearchLayer(index, enter_point, 1, i)[0];
    }
    for (int i = std::min(max_level_, level); i >= 0; --i)
    {
        if (enter_point == index)
        {
            enter_point = enter_points_[i];
            if (enter_point == index)
            {
                continue;
            }
        }
        auto nearest_neighbors = SearchLayer(index, enter_point, ef_construction_, i);
        enter_point = nearest_neighbors[0];
        if (nearest_neighbors.size() > M_)
        {
            nearest_neighbors.resize(M_);
        }
        layers_[i].graph_[index] = nearest_neighbors;
        for (size_t next : nearest_neighbors)
        {
            layers_[i].graph_[next].push_back(index);
            size_t maxM = maxM_;
            if (i == 0)
            {
                maxM = maxM0_;
            }
            if (layers_[i].graph_[next].size() > maxM)
            {
                layers_[i].graph_[next] = SelectNeighboursSimple(next, layers_[i].graph_[next], maxM);
            }
        }
    }
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level)
{
    Layer &layer = layers_[level];
    ++current_was_;
    was_[enter_point]=current_was_;
    std::priority_queue<std::pair<float, size_t>,
                        std::vector<std::pair<float, size_t>>,
                        std::greater<std::pair<float, size_t>>>
        candidates;
    candidates.emplace(space_.Distance(data_[enter_point], data_[query]), enter_point);
    std::priority_queue<std::pair<float, size_t>,
                        std::vector<std::pair<float, size_t>>,
                        std::less<std::pair<float, size_t>>>
        nearest_neighbours;
    nearest_neighbours.emplace(space_.Distance(data_[enter_point], data_[query]), enter_point);

    while (!candidates.empty())
    {
        size_t current = candidates.top().second;
        candidates.pop();
        size_t furthest = nearest_neighbours.top().second;
        if (space_.Distance(data_[current], data_[query]) > space_.Distance(data_[furthest], data_[query]))
        {
            break;
        }
        for (size_t next : layer.graph_[current])
        {
            if (was_[next]!=current_was_)
            {
                was_[next]=current_was_;
                furthest = nearest_neighbours.top().second;
                FloatType distance = space_.Distance(data_[next], data_[query]);
                if (distance < space_.Distance(data_[furthest], data_[query]) or nearest_neighbours.size() < ef)
                {
                    candidates.emplace(distance, next);
                    nearest_neighbours.emplace(distance, next);
                    if (nearest_neighbours.size() > ef)
                    {
                        nearest_neighbours.pop();
                    }
                }
            }
        }
    }
    std::vector<size_t> nearest_neighbours_vector;
    while (!nearest_neighbours.empty())
    {
        nearest_neighbours_vector.push_back(nearest_neighbours.top().second);
        nearest_neighbours.pop();
    }
    std::reverse(nearest_neighbours_vector.begin(), nearest_neighbours_vector.end());
    return nearest_neighbours_vector;
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::SelectNeighboursSimple(size_t query, std::vector<size_t> candidates, size_t M)
{
    std::vector<std::pair<FloatType, size_t>> array;
    array.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        array.emplace_back(space_.Distance(data_[query], data_[candidates[i]]), candidates[i]);
    }
    std::sort(array.begin(), array.end());
    std::vector<size_t> nearest_neighbours;
    nearest_neighbours.reserve(candidates.size());
    for (size_t i = 0; i < std::min(M, array.size()); ++i)
    {
        nearest_neighbours.push_back(array[i].second);
    }
    return nearest_neighbours;
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::Search(size_t query, size_t K, size_t ef)
{
    size_t enter_point = enter_points_[max_level_];
    for (size_t i = max_level_; i >= 1; --i)
    {
        enter_point = SearchLayer(query, enter_point, 1, i)[0];
    }
    auto nearest_neighbours = SearchLayer(query, enter_point, ef, 0);
    if (nearest_neighbours.size() > K)
    {
        nearest_neighbours.resize(K);
    }
    return nearest_neighbours;
}
