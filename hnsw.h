#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <iostream>

#include "primitives.h"

typedef std::priority_queue<std::pair<float, size_t>,
                            std::vector<std::pair<float, size_t>>,
                            std::less<std::pair<float, size_t>>>
    QueueLess;

typedef std::priority_queue<std::pair<float, size_t>,
                            std::vector<std::pair<float, size_t>>,
                            std::greater<std::pair<float, size_t>>>
    QueueGreater;

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
        data_.reserve(max_elements_);
    }
    void Add(const Point &point, int level);
    QueueLess SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level);
    std::vector<size_t> SelectNeighbours(size_t query, QueueLess &candidates, size_t M);
    std::vector<size_t> Search(size_t query, size_t K, size_t ef);
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;
    size_t enter_point_;
    size_t size_ = 0;
    int max_level_ = -1;
    std::vector<Point> data_;
    std::vector<Layer> layers_;
    Space space_;
    std::vector<size_t> was_;
    size_t current_was_;
    size_t max_elements_;
};

template <typename Space>
void HNSW<Space>::Add(const Point &point, int level)
{
    data_.push_back(point);
    size_t index = size_;
    ++size_;
    size_t enter_point = enter_point_;
    for (int i = max_level_; i > level; --i)
    {
        enter_point = SearchLayer(index, enter_point, 1, i).top().second;
    }
    for (int i = std::min(max_level_, level); i >= 0; --i)
    {
        auto nearest_neighbors = SearchLayer(index, enter_point, ef_construction_, i);
        layers_[i].graph_[index] = SelectNeighbours(index, nearest_neighbors, M_);
        enter_point = layers_[i].graph_[index][0];

        for (size_t next : layers_[i].graph_[index])
        {
            layers_[i].graph_[next].push_back(index);
            size_t maxM = maxM_;
            if (i == 0)
            {
                maxM = maxM0_;
            }
            if (layers_[i].graph_[next].size() > maxM)
            {
                std::priority_queue<std::pair<float, size_t>,
                                    std::vector<std::pair<float, size_t>>,
                                    std::less<std::pair<float, size_t>>>
                    queue;
                for (size_t neighbour : layers_[i].graph_[next])
                {
                    queue.emplace(space_.Distance(data_[next], data_[neighbour]), neighbour);
                }
                layers_[i].graph_[next] = SelectNeighbours(next, queue, maxM);
            }
        }
    }
    if (level > max_level_)
    {
        for (int i = max_level_ + 1; i <= level; ++i)
        {
            layers_.push_back(Layer());
            layers_.back().graph_.reserve(max_elements_);
        }
        max_level_ = level;
        enter_point_ = size_ - 1;
    }
}

template <typename Space>
inline QueueLess HNSW<Space>::SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level)
{
    Layer &layer = layers_[level];
    ++current_was_;
    was_[enter_point] = current_was_;
    QueueGreater candidates;
    FloatType enter_point_distance = space_.Distance(data_[enter_point], data_[query]);
    candidates.emplace(enter_point_distance, enter_point);
    QueueLess nearest_neighbours;
    nearest_neighbours.emplace(enter_point_distance, enter_point);
    while (!candidates.empty())
    {
        size_t current = candidates.top().second;
        candidates.pop();
        FloatType furthest_distance = nearest_neighbours.top().first;
        if (space_.Distance(data_[current], data_[query]) > furthest_distance and
            nearest_neighbours.size() == ef)
        {
            break;
        }
        for (size_t next : layer.graph_[current])
        {
            if (was_[next] != current_was_)
            {
                was_[next] = current_was_;
                furthest_distance = nearest_neighbours.top().first;
                FloatType distance = space_.Distance(data_[next], data_[query]);
                if (distance < furthest_distance or nearest_neighbours.size() < ef)
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
    return nearest_neighbours;
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::SelectNeighbours(size_t query, std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, std::less<std::pair<float, size_t>>> &candidates, size_t M)
{
    if (candidates.size() < M)
    {
        std::vector<size_t> array;
        array.reserve(maxM0_);
        while (!candidates.empty())
        {
            array.push_back(candidates.top().second);
            candidates.pop();
        }
        std::reverse(array.begin(), array.end());
        return array;
    }
    std::vector<std::pair<float, size_t>> queue;
    queue.reserve(candidates.size());
    while (!candidates.empty())
    {
        queue.push_back(candidates.top());
        candidates.pop();
    }
    std::reverse(queue.begin(), queue.end());
    std::vector<size_t> selected;
    selected.reserve(maxM0_);
    for (auto &[dist, element] : queue)
    {
        if (selected.size() >= M)
        {
            break;
        }
        bool good = true;
        for (size_t neighbour : selected)
        {
            FloatType curdist = space_.Distance(data_[element], data_[neighbour]);
            if (curdist < dist)
            {
                good = false;
            }
        }
        if (good)
        {
            selected.push_back(element);
        }
    }
    return selected;
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::Search(size_t query, size_t K, size_t ef)
{
    size_t enter_point = enter_point_;
    for (size_t i = max_level_; i >= 1; --i)
    {
        enter_point = SearchLayer(query, enter_point, 1, i).top().second;
    }
    auto nearest_neighbours = SearchLayer(query, enter_point, ef, 0);
    while (nearest_neighbours.size() > K)
    {
        nearest_neighbours.pop();
    }
    std::vector<size_t> array;
    array.reserve(nearest_neighbours.size());
    while (!nearest_neighbours.empty())
    {
        array.push_back(nearest_neighbours.top().second);
        nearest_neighbours.pop();
    }
    std::reverse(array.begin(), array.end());
    return array;
}
