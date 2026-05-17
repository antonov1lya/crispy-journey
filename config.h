#pragma once

// SELECT DATASET
#define SIFT1M
// #define GLOVE100
// #define GIST1M
// #define DEEP1B
// #define FASHION_MNIST

// SELECT TASK
// #define CREATE_TASK
#define BENCHMARK_TASK
// #define REORDER_TASK

// REORDERING TYPE FOR REORDER_TASK
#define REORDERING_TYPE_LOCAL_SEARCH
// #define REORDERING_TYPE_BFS
// #define REORDERING_TYPE_MST

// FILE NAMES
#define CREATE_TASK_INDEX_FILE "/base.bin"
#define BENCHMARK_TASK_INDEX_FILE "/base.bin"
#define BENCHMARK_TASK_LOG_FILE "/base.csv"
#define REORDERING_TASK_INDEX_FILE "/reordering.bin"

#define REORDER

#ifdef SIFT1M
#define DATASET "sift1m"
#define SIZE 128
#define SPACE SpaceL2
#endif

#ifdef GLOVE100
#define DATASET "glove100"
#define SIZE 100
#define SPACE SpaceCosine
#endif

#ifdef GIST1M
#define DATASET "gist1m"
#define SIZE 960
#define SPACE SpaceL2
#endif

#ifdef DEEP1B
#define DATASET "deep1b"
#define SIZE 96
#define SPACE SpaceCosine
#endif

#ifdef FASHION_MNIST
#define DATASET "fashion_mnist"
#define SIZE 784
#define SPACE SpaceL2
#endif

#define BITS 256

#define SUBSPACES 32
#define SUBSIZE (SIZE / SUBSPACES)
// #define PQ
// #define BATCH

// #define FIND_EF
#define EF_R 2000
