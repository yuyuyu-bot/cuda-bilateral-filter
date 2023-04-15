cmake_minimum_required(VERSION 3.21)
project(bilateral_filter LANGUAGES CXX CUDA)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV version: " ${OpenCV_VERSION})
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# source files
file(GLOB src ${CMAKE_SOURCE_DIR}/src/*.c*)

# include directories
set(inc
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# libraries
set(lib
    ${OpenCV_LIBRARIES}
)

add_executable(${PROJECT_NAME} ${src})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -Wall)
target_include_directories(${PROJECT_NAME} PRIVATE ${inc})
target_link_libraries(${PROJECT_NAME} PRIVATE ${lib})
