
cmake_minimum_required (VERSION 3.0)
project (FNN)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mssse3 -msse4.2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB_RECURSE SOURCE "Source/*.cpp")
file(GLOB_RECURSE HEADERS "Headers/*.h")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/staticlib)


find_package (Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)


#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()


include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/Headers)

add_executable (FNN ${SOURCE} ${HEADERS} )
#target_link_libraries (FNN Eigen3::Eigen)
target_link_libraries(FNN PRIVATE OpenMP::OpenMP_CXX)
