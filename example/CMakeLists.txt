set(LIB_PATH "/home/eric/Desktop/libtorch-video-classification/third_path/")

#cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(${CUDA_INCLUDE_DIRS})
#opencv
set(OpenCV_DIR "/home/eric/Desktop/libtorch-video-classification/third_path/opencv/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
message("OpenCV_INCLUDE_DIRS" ${OpenCV_INCLUDE_DIRS})
#trtorch
#set(TRTorch_INCLUDE_DIR "/home/eric/Desktop/libtorch-video-classification/third_path/trtorch/include/")
#include_directories(${TRTorch_INCLUDE_DIR})
#set(TRTorch_LIBS "/home/eric/Desktop/libtorch-video-classification/third_path/trtorch/lib/libtrtorch.so")

include_directories(${PROJECT_SOURCE_DIR}/include)
set(EXAMPLE_BIN_PATH ${CMAKE_BINARY_DIR}/)

set(DEP_LIBS DetectorModule)

add_executable(test_detect test_detector.cpp)
set_target_properties(test_detect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_detect ${DEP_LIBS})

