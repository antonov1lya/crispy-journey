cmake_minimum_required(VERSION 3.5.0)
project(anns VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(anns main.cpp primitives.h hnsw.h)
