#pragma once

// #define SIFT1M
// #define GLOVE100
#define GIST1M
// #define DEEP1B

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

#define BITS 256

#define SUBSPACES 32
#define SUBSIZE (SIZE / SUBSPACES)
#define PQ
#define BATCH

// #define FIND_EF
#define EF_R 600
