#pragma once

#include <cmath>
#include <queue>
#include <vector>

typedef float FloatType;
typedef int IntType;

typedef std::priority_queue<std::pair<FloatType, IntType>,
                            std::vector<std::pair<FloatType, IntType>>,
                            std::less<std::pair<FloatType, IntType>>>
    QueueLess;

typedef std::priority_queue<std::pair<FloatType, IntType>,
                            std::vector<std::pair<FloatType, IntType>>,
                            std::greater<std::pair<FloatType, IntType>>>
    QueueGreater;

#define REORDER

#define ALIGN64 64
#define ALIGN4 4

// #define SIZE 128
// #define SIZE 960
#define SIZE 784
// #define SIZE 100

struct SpaceL2 {
    FloatType Distance(FloatType* x, FloatType* y) {
        ++computations_;
        FloatType distance = 0;
        for (IntType i = 0; i < SIZE; ++i) {
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
    FloatType Distance(FloatType* x, FloatType* y) {
        ++computations_;
        FloatType distance = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            distance += x[i] * y[i];
        }
        return 1 - distance;
    }
    IntType GetComputationsNumber() {
        return computations_;
    }
    void FlushComputationsNumber() {
        computations_ = 0;
    }
    IntType computations_ = 0;
};

void Normalize(FloatType* x) {
    FloatType norm = 0;
    for (IntType i = 0; i < SIZE; ++i) {
        norm += x[i] * x[i];
    }
    norm = std::sqrt(norm);
    for (IntType i = 0; i < SIZE; ++i) {
        x[i] /= norm;
    }
}
