#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "primitives.h"

typedef std::priority_queue<std::pair<FloatType, size_t>,
                            std::vector<std::pair<FloatType, size_t>>,
                            std::less<std::pair<FloatType, size_t>>>
    QueueLess;

typedef std::priority_queue<std::pair<FloatType, size_t>,
                            std::vector<std::pair<FloatType, size_t>>,
                            std::greater<std::pair<FloatType, size_t>>>
    QueueGreater;

struct Node
{
    Node(size_t max_level)
    {
        neighbors_.resize(max_level + 1);
    }
    std::vector<std::vector<size_t>> neighbors_;
};

template <typename Space>
struct HNSW
{
    HNSW(size_t M, size_t ef_construction, size_t max_elements) : M_{M}, maxM_{M}, maxM0_{2 * M},
                                                                  ef_construction_{ef_construction}, max_elements_{max_elements}
    {
        was_ = std::vector<size_t>(max_elements_);
        data_.reserve(max_elements_);
        graph_.reserve(max_elements_);
    }
    void Add(const Point &point, int level);
    QueueLess SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level);
    std::vector<size_t> SelectNeighbours(size_t query, QueueLess &candidates, size_t M, size_t maxM);
    std::vector<size_t> Search(size_t query, size_t K, size_t ef);
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;
    size_t enter_point_;
    size_t size_ = 0;
    size_t current_was_;
    size_t max_elements_;
    int max_level_ = -1;
    std::vector<Point> data_;
    std::vector<size_t> was_;
    std::vector<Node> graph_;
    Space space_;
};

template <typename Space>
void HNSW<Space>::Add(const Point &point, int level)
{
    data_.push_back(point);
    graph_.push_back(Node(level));
    size_t index = size_++;
    size_t enter_point = enter_point_;
    for (int i = max_level_; i > level; --i)
    {
        enter_point = SearchLayer(index, enter_point, 1, i).top().second;
    }
    for (int i = std::min(max_level_, level); i >= 0; --i)
    {
        size_t maxM = maxM_;
        if (i == 0)
        {
            maxM = maxM0_;
        }
        auto nearest_neighbors = SearchLayer(index, enter_point, ef_construction_, i);
        graph_[index].neighbors_[i] = SelectNeighbours(index, nearest_neighbors, M_, maxM);
        enter_point = graph_[index].neighbors_[i][0];
        for (size_t next : graph_[index].neighbors_[i])
        {
            graph_[next].neighbors_[i].push_back(index);
            if (graph_[next].neighbors_[i].size() > maxM)
            {
                QueueLess queue;
                for (size_t neighbour : graph_[next].neighbors_[i])
                {
                    queue.emplace(space_.Distance(data_[next], data_[neighbour]), neighbour);
                }
                graph_[next].neighbors_[i] = SelectNeighbours(next, queue, maxM, maxM);
            }
        }
    }
    if (level > max_level_)
    {
        max_level_ = level;
        enter_point_ = index;
    }
}

template <typename Space>
inline QueueLess HNSW<Space>::SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level)
{
    was_[enter_point] = ++current_was_;
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
        for (size_t next : graph_[current].neighbors_[level])
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
inline std::vector<size_t> HNSW<Space>::SelectNeighbours(size_t query, QueueLess &candidates, size_t M, size_t maxM)
{
    if (candidates.size() < M)
    {
        std::vector<size_t> array;
        array.reserve(maxM + 1);
        while (!candidates.empty())
        {
            array.push_back(candidates.top().second);
            candidates.pop();
        }
        std::reverse(array.begin(), array.end());
        return array;
    }
    std::vector<std::pair<FloatType, size_t>> queue;
    queue.reserve(candidates.size());
    while (!candidates.empty())
    {
        queue.push_back(candidates.top());
        candidates.pop();
    }
    std::reverse(queue.begin(), queue.end());
    std::vector<size_t> selected;
    selected.reserve(maxM + 1);
    for (auto &[distance, element] : queue)
    {
        if (selected.size() >= M)
        {
            break;
        }
        bool good = true;
        for (size_t neighbour : selected)
        {
            FloatType cur_distance = space_.Distance(data_[element], data_[neighbour]);
            if (cur_distance < distance)
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
