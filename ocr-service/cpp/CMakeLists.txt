cmake_minimum_required(VERSION 3.16)
project(ocr_preprocess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ocr_preprocess preprocess.cpp)
target_link_libraries(ocr_preprocess ${OpenCV_LIBS})

install(TARGETS ocr_preprocess RUNTIME DESTINATION bin)
