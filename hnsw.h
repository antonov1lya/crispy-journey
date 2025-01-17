#pragma once

#include <vector>
#include <unordered_map>
#include <queue>

#include "primitives.h"

struct Layer
{
    explicit Layer(bool bottom) : bottom_{bottom}
    {
    }
    void Add(size_t index);
    // convert global index to layer index
    size_t Encoder(size_t index) const;
    // convert layer index to layer global
    size_t Decoder(size_t index) const;
    std::vector<std::vector<size_t>> graph_;
    std::unordered_map<size_t, size_t> encoder_;
    std::unordered_map<size_t, size_t> decoder_;
    size_t size_ = 0;
    bool bottom_;
};

template <typename Space>
struct HNSW
{
    HNSW(size_t M, size_t ef_construction) : M_{M}, maxM_{M}, maxM0_{2 * M},
                                             ef_construction_{ef_construction}
    {
    }
    void Add(const Point &point, int level);
    std::vector<size_t> SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level);
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;
    size_t enter_point_ = 0;
    size_t size_ = 0;
    int max_level_ = -1;
    std::vector<Point> data_;
    std::vector<Layer> layers_;
};

template <typename Space>
void HNSW<Space>::Add(const Point &point, int level)
{
    data_.push_back(point);
    size_t index = size_;
    size_++;
    size_t enter_point = enter_point_;
    if (level > max_level_)
    {
        for (int i = max_level_ + 1; i <= level; ++i)
        {
            if (i == 0)
            {
                layers_.push_back(Layer(true));
            }
            else
            {
                layers_.push_back(Layer(false));
            }
            layers_[i].Add(index);
        }
        enter_point_ = index;
        std::swap(level, max_level_);
    }
    // TODO
}

template <typename Space>
inline std::vector<size_t> HNSW<Space>::SearchLayer(size_t query, size_t enter_point, size_t ef, size_t level)
{
    return std::vector<size_t>();
}
