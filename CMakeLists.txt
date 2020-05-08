project(opencv_detector)
cmake_minimum_required(VERSION 3.16)

find_package(Boost REQUIRED COMPONENTS filesystem)
add_subdirectory(opencv_dnn)
add_executable(opencv_detector main.cpp)

target_include_directories(opencv_detector PUBLIC ${ultraface_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(opencv_detector ultraface ${Boost_LIBRARIES})
