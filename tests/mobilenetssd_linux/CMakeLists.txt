
# aux_source_directory(./ SRC_LIST)
set(SRC_LIST ./mobilenetssd.cpp)

add_library(mnssd_linux SHARED ${SRC_LIST})

# 生成可执行文件
add_executable(mnssd mobilenetssd.cpp)

# 链接库文件
target_link_libraries(mnssd_linux opencv_highgui opencv_imgproc opencv_core jpeg pthread dl)