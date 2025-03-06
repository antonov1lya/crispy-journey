#pragma once

#include <vector>

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
    void Normalize()
    {
        FloatType norm = 0;
        for (size_t i = 0; i < data_.size(); ++i)
        {
            norm += data_[i] * data_[i];
        }
        norm = sqrt(norm);
        for (size_t i = 0; i < data_.size(); ++i)
        {
            data_[i] /= norm;
        }
    }
    std::vector<FloatType> data_;
};

Point operator-(Point &a, Point &b)
{
    Point result(a.Size());
    for (size_t i = 0; i < a.Size(); ++i)
    {
        result[i] = a[i] - b[i];
    }
    return result;
}

struct SpaceL2
{
    FloatType Distance(const struct Point &x, const struct Point &y)
    {
        ++computations_;
        FloatType distance = 0;
        for (size_t i = 0; i < x.Size(); ++i)
        {
            FloatType diff = x[i] - y[i];
            distance += diff * diff;
        }
        return distance;
    }
    FloatType Cos(const struct Point &x, const struct Point &y)
    {
        FloatType result = 0;
        for (size_t i = 0; i < x.Size(); ++i)
        {
            result += x[i] * y[i];
        }
        return result;
    }
    size_t GetComputationsNumber()
    {
        return computations_;
    }
    void FlushComputationsNumber()
    {
        computations_ = 0;
    }
    size_t computations_ = 0;
};
