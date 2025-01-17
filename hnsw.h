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
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;
    std::vector<Point> data_;
    std::vector<Layer> layers_;
};