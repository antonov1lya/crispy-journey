#pragma once

#include <string.h>

#include <cmath>
#include <queue>
#include <vector>

#include "config.h"

using FloatType = float;
using IntType = int;

using QueueLess =
    std::priority_queue<std::pair<FloatType, IntType>, std::vector<std::pair<FloatType, IntType>>,
                        std::less<std::pair<FloatType, IntType>>>;
using QueueGreater =
    std::priority_queue<std::pair<FloatType, IntType>, std::vector<std::pair<FloatType, IntType>>,
                        std::greater<std::pair<FloatType, IntType>>>;

struct SpaceL2 {
    FloatType Distance(const FloatType* x, const FloatType* y) {
        ++computations_;
        FloatType distance = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            FloatType diff = x[i] - y[i];
            distance += diff * diff;
        }
        return distance;
    }
    FloatType DistanceSubspace(const FloatType* x, const FloatType* y) {
        FloatType distance = 0;
        for (IntType i = 0; i < SUBSIZE; ++i) {
            FloatType diff = x[i] - y[i];
            distance += diff * diff;
        }
        return distance;
    }
    IntType GetComputationsNumber() {
        return computations_;
    }
    void FlushComputationsNumber() {
        computations_ = 0;
    }
    IntType computations_ = 0;
};

struct SpaceCosine {
    FloatType Distance(const FloatType* x, const FloatType* y) {
        // NOTE: works only with normalized vectors
        ++computations_;
        FloatType distance = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            distance += x[i] * y[i];
        }
        return 1 - distance;
    }
    FloatType DistanceSubspace(const FloatType* x, const FloatType* y) {
        FloatType distance = 0;
        for (IntType i = 0; i < SUBSIZE; ++i) {
            FloatType diff = x[i] - y[i];
            distance += diff * diff;
        }
        return distance;
    }
    IntType GetComputationsNumber() {
        return computations_;
    }
    void FlushComputationsNumber() {
        computations_ = 0;
    }
    IntType computations_ = 0;
};

struct Point {
    std::vector<FloatType> data;
    void Normalize() {
        FloatType norm = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            norm += data[i] * data[i];
        }
        norm = std::sqrt(norm);
        for (IntType i = 0; i < SIZE; ++i) {
            data[i] /= norm;
        }
    }
    Point(FloatType* start, FloatType* end) {
        data.resize(SIZE);
        for (IntType i = 0; i < SIZE; ++i) {
            data[i] = end[i] - start[i];
        }
        Normalize();
    }
};

FloatType CalcCos(const Point x, const Point& y) {
    FloatType distance = 0;
    for (IntType i = 0; i < SIZE; ++i) {
        distance += x.data[i] * y.data[i];
    }
    return distance;
}

void MatVecMul(const FloatType* __restrict matrix, const FloatType* __restrict vector,
               FloatType* __restrict result) {
    for (IntType i = 0; i < SIZE; ++i) {
        const FloatType* row = matrix + i * SIZE;
        FloatType sum = 0;
        for (IntType j = 0; j < SIZE; ++j) {
            sum += row[j] * vector[j];
        }
        result[i] = sum;
    }
}
