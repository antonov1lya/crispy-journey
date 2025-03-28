#pragma once

#include <vector>

typedef float FloatType;
typedef int IntType;

#define SIZE 128

struct Point
{
    explicit Point(IntType n)
    {
        data_.resize(n);
    }
    IntType Size() const
    {
        return SIZE;
    }
    FloatType &operator[](IntType i)
    {
        return data_[i];
    }
    FloatType operator[](IntType i) const
    {
        return data_[i];
    }
    void Normalize()
    {
        FloatType norm = 0;
        for (IntType i = 0; i < data_.size(); ++i)
        {
            norm += data_[i] * data_[i];
        }
        norm = sqrt(norm);
        for (IntType i = 0; i < data_.size(); ++i)
        {
            data_[i] /= norm;
        }
    }
    std::vector<FloatType> data_;
};

Point operator-(Point &a, Point &b)
{
    Point result(a.Size());
    for (IntType i = 0; i < a.Size(); ++i)
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
        for (IntType i = 0; i < SIZE; ++i)
        {
            FloatType diff = x[i] - y[i];
            distance += diff * diff;
        }
        return distance;
    }
    FloatType Cos(const struct Point &x, const struct Point &y)
    {
        FloatType result = 0;
        for (IntType i = 0; i < x.Size(); ++i)
        {
            result += x[i] * y[i];
        }
        return result;
    }
    IntType GetComputationsNumber()
    {
        return computations_;
    }
    void FlushComputationsNumber()
    {
        computations_ = 0;
    }
    size_t computations_ = 0;
};
