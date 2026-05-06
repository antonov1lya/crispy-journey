#pragma once

#include <string.h>
#include <sys/mman.h>

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
    void DistanceBatch2(const FloatType* q, const FloatType* x1, const FloatType* x2, FloatType* r1,
                        FloatType* r2) {
        computations_ += 2;
        FloatType d1 = 0, d2 = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            FloatType diff1 = q[i] - x1[i];
            FloatType diff2 = q[i] - x2[i];
            d1 += diff1 * diff1;
            d2 += diff2 * diff2;
        }
        *r1 = d1;
        *r2 = d2;
    }
    void DistanceBatch4(const FloatType* q, const FloatType* x1, const FloatType* x2,
                        const FloatType* x3, const FloatType* x4, FloatType* r1, FloatType* r2,
                        FloatType* r3, FloatType* r4) {
        computations_ += 4;
        FloatType d1 = 0, d2 = 0, d3 = 0, d4 = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            FloatType diff1 = q[i] - x1[i];
            FloatType diff2 = q[i] - x2[i];
            FloatType diff3 = q[i] - x3[i];
            FloatType diff4 = q[i] - x4[i];

            d1 += diff1 * diff1;
            d2 += diff2 * diff2;
            d3 += diff3 * diff3;
            d4 += diff4 * diff4;
        }

        *r1 = d1;
        *r2 = d2;
        *r3 = d3;
        *r4 = d4;
    }
    void DistanceBatch8(const FloatType* q, const FloatType* x1, const FloatType* x2,
                        const FloatType* x3, const FloatType* x4, const FloatType* x5,
                        const FloatType* x6, const FloatType* x7, const FloatType* x8,
                        FloatType* r1, FloatType* r2, FloatType* r3, FloatType* r4, FloatType* r5,
                        FloatType* r6, FloatType* r7, FloatType* r8) {
        computations_ += 8;
        FloatType d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0, d8 = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            FloatType diff1 = q[i] - x1[i];
            FloatType diff2 = q[i] - x2[i];
            FloatType diff3 = q[i] - x3[i];
            FloatType diff4 = q[i] - x4[i];
            FloatType diff5 = q[i] - x5[i];
            FloatType diff6 = q[i] - x6[i];
            FloatType diff7 = q[i] - x7[i];
            FloatType diff8 = q[i] - x8[i];

            d1 += diff1 * diff1;
            d2 += diff2 * diff2;
            d3 += diff3 * diff3;
            d4 += diff4 * diff4;
            d5 += diff5 * diff5;
            d6 += diff6 * diff6;
            d7 += diff7 * diff7;
            d8 += diff8 * diff8;
        }

        *r1 = d1;
        *r2 = d2;
        *r3 = d3;
        *r4 = d4;
        *r5 = d5;
        *r6 = d6;
        *r7 = d7;
        *r8 = d8;
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
    void DistanceBatch2(const FloatType* q, const FloatType* x1, const FloatType* x2, FloatType* r1,
                        FloatType* r2) {
        computations_ += 2;
        FloatType d1 = 0, d2 = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            d1 += q[i] * x1[i];
            d2 += q[i] * x2[i];
        }
        *r1 = 1 - d1;
        *r2 = 1 - d2;
    }
    void DistanceBatch4(const FloatType* q, const FloatType* x1, const FloatType* x2,
                        const FloatType* x3, const FloatType* x4, FloatType* r1, FloatType* r2,
                        FloatType* r3, FloatType* r4) {
        computations_ += 4;
        FloatType d1 = 0, d2 = 0, d3 = 0, d4 = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            d1 += q[i] * x1[i];
            d2 += q[i] * x2[i];
            d3 += q[i] * x3[i];
            d4 += q[i] * x4[i];
        }

        *r1 = 1 - d1;
        *r2 = 1 - d2;
        *r3 = 1 - d3;
        *r4 = 1 - d4;
    }
    void DistanceBatch8(const FloatType* q, const FloatType* x1, const FloatType* x2,
                        const FloatType* x3, const FloatType* x4, const FloatType* x5,
                        const FloatType* x6, const FloatType* x7, const FloatType* x8,
                        FloatType* r1, FloatType* r2, FloatType* r3, FloatType* r4, FloatType* r5,
                        FloatType* r6, FloatType* r7, FloatType* r8) {
        computations_ += 8;
        FloatType d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0, d8 = 0;
        for (IntType i = 0; i < SIZE; ++i) {
            d1 += q[i] * x1[i];
            d2 += q[i] * x2[i];
            d3 += q[i] * x3[i];
            d4 += q[i] * x4[i];
            d5 += q[i] * x5[i];
            d6 += q[i] * x6[i];
            d7 += q[i] * x7[i];
            d8 += q[i] * x8[i];
        }

        *r1 = 1 - d1;
        *r2 = 1 - d2;
        *r3 = 1 - d3;
        *r4 = 1 - d4;
        *r5 = 1 - d5;
        *r6 = 1 - d6;
        *r7 = 1 - d7;
        *r8 = 1 - d8;
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

void* HugeAlloc(size_t total_size_bytes) {
    size_t hugepage_size = 2 * 1024 * 1024;
    size_t size = (total_size_bytes + hugepage_size - 1) & ~(hugepage_size - 1);
    void* ptr =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    std::cout << ptr << "\n";
    return ptr;
}

void HugeDealloc(void* ptr, size_t total_size_bytes) {
    size_t hugepage_size = 2 * 1024 * 1024;
    size_t size = (total_size_bytes + hugepage_size - 1) & ~(hugepage_size - 1);
    munmap(ptr, size);
}

#define ALIGN64 64
void* MyAlignedAlloc(size_t size) {
    size_t aligned_size = (size + ALIGN64 - 1) & ~(ALIGN64 - 1);
    void* ptr = aligned_alloc(ALIGN64, aligned_size);
    return ptr;
}
