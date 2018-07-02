# Copyright (C) 2018 deeplearningrobotics.ai
cmake_minimum_required(VERSION 2.8.12)
project(tensor_swarm_bot)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(PkgConfig)
pkg_check_modules(ARGOS REQUIRED argos3_simulator)
set(ARGOS_PREFIX ${ARGOS_PREFIX} CACHE INTERNAL "")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ARGOS_PREFIX}/share/argos3/cmake)

include(ARGoSCheckQTOpenGL)

find_package(GSL REQUIRED)
find_package(Lua52 REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} ${LUA_INCLUDE_DIR})

link_directories(${ARGOS_LIBRARY_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/controllers)
add_subdirectory(tensor_swarm_bot)
