cmake_minimum_required(VERSION 3.18)
project(TwinTubeOCR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPTONICA REQUIRED lept)

# Check for CUDA support
find_package(CUDA QUIET)
if(OpenCV_CUDA_VERSION OR CUDA_FOUND)
    message(STATUS "CUDA support: ENABLED")
    add_definitions(-DHAVE_CUDA)
else()
    message(STATUS "CUDA support: DISABLED")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TESSERACT_INCLUDE_DIRS}
    ${LEPTONICA_INCLUDE_DIRS}
)

# Main OCR executable
add_executable(twintube_ocr main.cpp)

target_link_libraries(twintube_ocr
    ${OpenCV_LIBS}
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    Threads::Threads
)

if(CUDA_FOUND)
    target_link_libraries(twintube_ocr ${CUDA_LIBRARIES})
endif()

# Benchmark executable
add_executable(ocr_benchmark benchmark.cpp)

target_link_libraries(ocr_benchmark
    ${OpenCV_LIBS}
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    Threads::Threads
)

# Legacy compatibility - keep 'main' as alias
add_executable(main main.cpp)
target_link_libraries(main
    ${OpenCV_LIBS}
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    Threads::Threads
)

# Install
install(TARGETS twintube_ocr ocr_benchmark main RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "TwinTube OCR Configuration:")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
