#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>
#include <fstream>
#include <iomanip>

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
    HNSW(std::ifstream &file);
    void Add(const Point &point, int level);
    QueueLess SearchLayer(Point &query, size_t enter_point, size_t ef, size_t level);
    QueueLess SearchLayer0(Point &query, size_t enter_point, size_t ef, size_t level);
    std::vector<size_t> SelectNeighbours(QueueLess &candidates, size_t M, size_t maxM);
    std::vector<size_t> Search(Point &query, size_t K, size_t ef);
    std::vector<size_t> SSG(size_t node, std::vector<size_t> &candidates, size_t M);

    void MemoryManager();
    void AddNeighborhood(size_t i, size_t j);
    void AddNeighborhood(size_t i);

    void Improve();
    void Save(std::ofstream &file, size_t precision = 1);
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;
    size_t enter_point_;
    size_t size_ = 0;
    size_t current_was_ = 0;
    size_t max_elements_;
    int max_level_ = -1;
    std::vector<Point> data_;
    std::vector<size_t> was_;
    std::vector<Node> graph_;

    std::vector<size_t> A0_;
    std::vector<size_t> B0_;
    size_t size0_ = 0;

    Space space_;
    float ssg_cos = 0.5;
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
        enter_point = SearchLayer(data_[index], enter_point, 1, i).top().second;
    }
    for (int i = std::min(max_level_, level); i >= 0; --i)
    {
        size_t maxM = maxM_;
        if (i == 0)
        {
            maxM = maxM0_;
        }
        auto nearest_neighbors = SearchLayer(data_[index], enter_point, ef_construction_, i);
        graph_[index].neighbors_[i] = SelectNeighbours(nearest_neighbors, M_, maxM);
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
                graph_[next].neighbors_[i] = SelectNeighbours(queue, maxM, maxM);
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
inline QueueLess HNSW<Space>::SearchLayer(Point &query, size_t enter_point, size_t ef, size_t level)
{
    was_[enter_point] = ++current_was_;
    QueueGreater candidates;
    FloatType enter_point_distance = space_.Distance(data_[enter_point], query);
    candidates.emplace(enter_point_distance, enter_point);
    QueueLess nearest_neighbours;
    nearest_neighbours.emplace(enter_point_distance, enter_point);
    while (!candidates.empty())
    {
        size_t current = candidates.top().second;
        candidates.pop();
        FloatType furthest_distance = nearest_neighbours.top().first;
        if (space_.Distance(data_[current], query) > furthest_distance and
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
                FloatType distance = space_.Distance(data_[next], query);
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
inline QueueLess HNSW<Space>::SearchLayer0(Point &query, size_t enter_point, size_t ef, size_t level)
{
    was_[enter_point] = ++current_was_;
    QueueGreater candidates;
    FloatType enter_point_distance = space_.Distance(data_[enter_point], query);
    candidates.emplace(enter_point_distance, enter_point);
    QueueLess nearest_neighbours;
    nearest_neighbours.emplace(enter_point_distance, enter_point);
    while (!candidates.empty())
    {
        size_t current = candidates.top().second;
        candidates.pop();
        FloatType furthest_distance = nearest_neighbours.top().first;
        if (space_.Distance(data_[current], query) > furthest_distance and
            nearest_neighbours.size() == ef)
        {
            break;
        }
        for (size_t i = B0_[2 * current]; i < B0_[2 * current + 1]; ++i)
        {
            size_t next = A0_[i];
            if (was_[next] != current_was_)
            {
                was_[next] = current_was_;
                furthest_distance = nearest_neighbours.top().first;
                FloatType distance = space_.Distance(data_[next], query);
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
inline std::vector<size_t> HNSW<Space>::SelectNeighbours(QueueLess &candidates, size_t M, size_t maxM)
{
    bool simple = true;
    if (simple)
    {
        std::vector<size_t> array;
        array.reserve(maxM + 1);
        while (!candidates.empty())
        {
            array.push_back(candidates.top().second);
            candidates.pop();
        }
        std::reverse(array.begin(), array.end());
        array.resize(std::min(M, array.size()));
        return array;
    }
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
inline std::vector<size_t> HNSW<Space>::SSG(size_t node, std::vector<size_t> &candidates, size_t M)
{

    std::sort(candidates.begin(), candidates.end());
    int new_size = std::unique(candidates.begin(), candidates.end()) - candidates.begin();
    candidates.resize(new_size);

    QueueLess q;
    for (size_t neighbour : candidates)
    {
        q.emplace(space_.Distance(data_[node], data_[neighbour]), neighbour);
    }

    std::vector<std::pair<FloatType, size_t>> queue;
    queue.reserve(q.size());
    while (!q.empty())
    {
        queue.push_back(q.top());
        q.pop();
    }
    std::reverse(queue.begin(), queue.end());
    std::vector<size_t> selected;
    std::vector<Point> dir;
    dir.reserve(M);
    selected.reserve(M + 1);
    for (auto &[distance, element] : queue)
    {
        if (selected.size() >= M)
        {
            break;
        }
        if (element == node)
            continue;
        Point curdir = data_[element] - data_[node];
        curdir.Normalize();
        bool good = true;
        for (size_t i = 0; i < selected.size(); ++i)
        {
            size_t neighbour = selected[i];
            FloatType cos = space_.Cos(dir[i], curdir);
            if (cos > ssg_cos)
            {
                good = false;
                break;
            }
        }
        if (good)
        {
            selected.push_back(element);
            dir.push_back(curdir);
        }
    }
    return selected;
}

template <typename Space>
inline void HNSW<Space>::MemoryManager()
{
    std::priority_queue<std::pair<size_t, std::pair<size_t, size_t>>,
                        std::vector<std::pair<size_t, std::pair<size_t, size_t>>>,
                        std::less<std::pair<size_t, std::pair<size_t, size_t>>>>
        q;
    B0_ = std::vector<size_t>(2 * size_);
    for (size_t i = 0; i < size_; ++i)
    {
        if (i % 10000 == 0)
        {
            std::cout << i << "\n";
        }
        std::set<int> candidates_;
        for (auto x : graph_[i].neighbors_[0])
        {
            if (i != x)
            {
                candidates_.insert(x);
            }
            // for (auto y : graph_[x].neighbors_[0])
            // {
            //     if (i != y)
            //     {
            //         candidates_.insert(y);
            //     }
            // }
        }

        std::set<int> cur;
        for (auto x : graph_[i].neighbors_[0])
        {
            cur.insert(x);
        }
        std::set<std::pair<size_t, std::pair<size_t, size_t>>> nq;
        for (auto x : candidates_)
        {
            int count = 0;
            for (auto y : graph_[x].neighbors_[0])
            {
                count += cur.count(y);
            }
            nq.insert({count, {i, x}});
            if (nq.size() > 5)
            {
                nq.erase(nq.begin());
            }
        }
        for (auto x : nq)
        {
            q.push(x);
        }
    }
    int cnt = 0;
    int num = 0;
    std::vector<int> was(size_, -1);
    while (!q.empty())
    {
        if (num >= 200000)
        {
            break;
        }
        auto t = q.top();
        q.pop();
        auto [i, j] = t.second;
        if (was[i] == -1 and was[j] == -1)
        {
            if (i > j)
            {
                std::swap(i, j);
            }
            was[i] = j;
            was[j] = j;
            cnt += t.first;
            num++;
        }
    }
    std::cout << cnt << "\n";
    int res = 0;
    for (int i = 0; i < size_; ++i)
    {
        if (was[i] == -1)
        {
            AddNeighborhood(i);
            res++;
        }
        else if (was[i] != i)
        {
            AddNeighborhood(i, was[i]);
        }
    }
    std::cout << res << "\n";
}

template <typename Space>
inline void HNSW<Space>::AddNeighborhood(size_t i, size_t j)
{
    std::set<size_t> i_minus_j;
    std::set<size_t> j_minus_i;
    std::set<size_t> intersection;
    for (size_t element : graph_[i].neighbors_[0])
    {
        i_minus_j.insert(element);
    }
    for (size_t element : graph_[j].neighbors_[0])
    {
        if (i_minus_j.count(element))
        {
            intersection.insert(element);
        }
        else
        {
            j_minus_i.insert(element);
        }
    }
    for (size_t element : intersection)
    {
        i_minus_j.erase(element);
    }
    B0_[2 * i] = size0_;
    for (auto x : i_minus_j)
    {
        A0_.push_back(x);
        size0_++;
    }
    B0_[2 * j] = size0_;
    for (auto x : intersection)
    {
        A0_.push_back(x);
        size0_++;
    }
    B0_[2 * i + 1] = size0_;
    for (auto x : j_minus_i)
    {
        A0_.push_back(x);
        size0_++;
    }
    B0_[2 * j + 1] = size0_;
}

template <typename Space>
inline void HNSW<Space>::AddNeighborhood(size_t i)
{
    B0_[2 * i] = size0_;
    for (auto x : graph_[i].neighbors_[0])
    {
        A0_.push_back(x);
        size0_++;
        // A0_[size0_++] = x;
    }
    B0_[2 * i + 1] = size0_;
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::Search(Point &query, size_t K, size_t ef)
{
    size_t enter_point = enter_point_;
    for (size_t i = max_level_; i >= 1; --i)
    {
        enter_point = SearchLayer(query, enter_point, 1, i).top().second;
    }
    auto nearest_neighbours = SearchLayer0(query, enter_point, ef, 0);
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

template <typename Space>
inline void HNSW<Space>::Improve()
{
    int l = 100, r = 50;
    std::vector<std::vector<size_t>> ne(size_);
    for (size_t node = 0; node < size_; ++node)
    {
        if (node % 10000 == 0)
        {
            std::cout << node << "\n";
        }
        for (size_t level = 0; level < 1; ++level)
        {
            std::vector<size_t> candidates;
            candidates.reserve(l);
            for (size_t x : graph_[node].neighbors_[level])
            {
                candidates.push_back(x);
                for (size_t y : graph_[x].neighbors_[level])
                {
                    candidates.push_back(y);
                }
            }
            ne[node] = SSG(node, candidates, r);
        }
    }
    for (size_t node = 0; node < size_; ++node)
    {
        graph_[node].neighbors_[0] = ne[node];
    }
    for (size_t node = 0; node < size_; ++node)
    {
        if (node % 10000 == 0)
        {
            std::cout << node << "\n";
        }
        for (size_t level = 0; level < 1; ++level)
        {
            for (size_t x : graph_[node].neighbors_[level])
            {
                graph_[x].neighbors_[level].push_back(node);
                graph_[x].neighbors_[level] = SSG(x, graph_[x].neighbors_[level], r);
            }
        }
    }
}

template <typename Space>
inline void HNSW<Space>::Save(std::ofstream &file, size_t precision)
{
    file << size_ << "\n";
    file << enter_point_ << "\n";
    file << M_ << "\n";
    file << ef_construction_ << "\n";
    file << max_level_ << "\n";
    size_t dim = data_[0].Size();
    file << dim << "\n";
    file << std::fixed << std::setprecision(precision);
    for (size_t node = 0; node < size_; ++node)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            file << data_[node][i] << " ";
        }
        file << "\n";
    }
    for (size_t node = 0; node < size_; ++node)
    {
        file << graph_[node].neighbors_.size() << '\n';
        for (size_t level = 0; level < graph_[node].neighbors_.size(); ++level)
        {
            file << graph_[node].neighbors_[level].size() << " ";
            for (size_t neighbour : graph_[node].neighbors_[level])
            {
                file << neighbour << " ";
            }
            file << "\n";
        }
    }
}

template <typename Space>
inline HNSW<Space>::HNSW(std::ifstream &file)
{
    file >> size_;
    graph_.reserve(size_);
    was_ = std::vector<size_t>(size_);
    max_elements_ = size_;
    file >> enter_point_;
    file >> M_;
    maxM_ = M_;
    maxM0_ = 2 * M_;
    file >> ef_construction_;
    file >> max_level_;
    size_t dim;
    file >> dim;
    data_ = std::vector<Point>(size_, Point(dim));
    for (size_t node = 0; node < size_; ++node)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            file >> data_[node][i];
        }
    }
    for (size_t node = 0; node < size_; ++node)
    {
        size_t level_number;
        file >> level_number;
        graph_.push_back(Node(level_number - 1));
        for (size_t level = 0; level < level_number; ++level)
        {
            if (level == 0)
            {
                graph_[node].neighbors_[level].reserve(maxM0_);
            }
            else
            {
                graph_[node].neighbors_[level].reserve(maxM_);
            }
            size_t neighbour_number;
            file >> neighbour_number;
            for (size_t it = 0; it < neighbour_number; ++it)
            {
                size_t neighbour;
                file >> neighbour;
                graph_[node].neighbors_[level].push_back(neighbour);
            }
        }
    }
}
