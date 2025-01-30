cmake_minimum_required(VERSION 3.5.0)
project(anns VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS  "-Ofast -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")

add_executable(anns main.cpp primitives.h hnsw.h)
