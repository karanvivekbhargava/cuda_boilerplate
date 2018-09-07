cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS imgcodecs highgui)

include_directories(
                    ${CMAKE_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS})

cuda_add_executable(image
                    ${headers}
                    src/image.cu)

target_link_libraries(image

                    ${OpenCV_LIBS})