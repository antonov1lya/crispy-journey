cmake_minimum_required(VERSION 3.5.0)
project(anns VERSION 0.1.0 LANGUAGES C CXX)

add_executable(anns main.cpp primitives.h hnsw.h)
