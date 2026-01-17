#pragma once

// #define SIFT1M
// #define GLOVE100
#define GIST1M

#define REORDER

#define SUBSPACES 32
// #define SUBSIZE 8
#define SUBSIZE 30
#define BITS 256

#define ALIGN64 64
#define ALIGN4 4

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
