cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(plane_extraction_based_localization)

find_package(PCL 1.2 REQUIRED)

include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (PEBL PEBL.cpp)
target_link_libraries (PEBL ${PCL_LIBRARIES})

