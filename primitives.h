#pragma once

#include <cmath>
#include <vector>

typedef float FloatType;
typedef int IntType;

// #define SIZE 960
#define SIZE 784

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
