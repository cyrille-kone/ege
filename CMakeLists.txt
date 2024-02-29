# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure # you pick up the best policies.
cmake_minimum_required(VERSION 3.25)
# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variable
project(aistats23
        DESCRIPTION "A bandit library")
set(CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR}/src/cpp)
add_executable(aistats23
        src/cpp/policies.hpp
        src/cpp/utils.hpp
        src/cpp/policies.cxx
        src/cpp/bandits.cxx
        src/cpp/bandits.hpp
        src/cpp/utils.cxx
        main.cxx)
target_link_libraries(aistats23)

