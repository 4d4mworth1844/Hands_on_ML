cmake_minimum_required(VERSION 3.27)
project(SuperTest_1_0_3)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(SuperTest_1_0_3
        ID_DSA_BTL1.cpp
        ID_DSA_BTL1.hpp
        main.hpp
        main_TestCase_1.cpp
        main_TestCase_2.cpp)
