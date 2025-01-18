#pragma once

#include <vector>
#include <cmath>

typedef float FloatType;

struct Point
{
    explicit Point(size_t n)
    {
        data_.resize(n);
    }
    size_t Size() const
    {
        return data_.size();
    }
    FloatType &operator[](size_t i)
    {
        return data_[i];
    }
    FloatType operator[](size_t i) const
    {
        return data_[i];
    }
    std::vector<FloatType> data_;
};

struct SpaceL2
{
    FloatType Distance(const struct Point &x, const struct Point &y) const
    {
        FloatType distance = 0;
        for (size_t i = 0; i < x.Size(); ++i)
        {
            distance += std::pow(x[i] - y[i], 2);
        }
        return distance;
    }
};
