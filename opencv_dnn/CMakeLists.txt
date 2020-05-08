project(ultraface)
cmake_minimum_required(VERSION 3.16)

set(OpenCV_DIR /home/ppotrykus/Downloads/opencv-3.4.9/build)
find_package(OpenCV 3.4.9 REQUIRED)

add_library(ultraface STATIC cv_dnn_ultraface.cpp)
target_include_directories(ultraface PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ultraface ${OpenCV_LIBS})
