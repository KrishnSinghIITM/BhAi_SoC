cmake_minimum_required(VERSION 3.22.1)
project("bhaisoc_native")

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(
    src/main/cpp
)

# Compile native dual-engine pipeline JNI library
add_library(bhaisoc_native SHARED
    src/main/cpp/native_pipeline.cpp
)

# Link against Android NDK core hardware and logging libraries
find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(bhaisoc_native
    ${log-lib}
    ${android-lib}
)
