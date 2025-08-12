cmake_minimum_required(VERSION 3.10)
project(HoughLaneDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HoughLaneDetector src/main.cpp)

target_link_libraries(HoughLaneDetector ${OpenCV_LIBS})
