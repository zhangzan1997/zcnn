project(ncnn)
cmake_minimum_required(VERSION 3.10)

# 设置变量
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -O3 -fpermissive -fopenmp -std=c++11")
# 设置输出库与可执行文件路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

# 寻找opencv路径
find_package(OpenCV REQUIRED)
# 添加opencv头文件
include_directories(${OpenCV_INCLUDE_DIRS})
message("opencv dir: ${OpenCV_INCLUDE_DIRS}")

# 添加头文件目录
message("================= ${CMAKE_SOURCE_DIR}")
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests)

# 添加库目录
link_directories(
    /usr/local/lib
    /usr/lib
)

# 添加子文件夹
add_subdirectory(./tests)





