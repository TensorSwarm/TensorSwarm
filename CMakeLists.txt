# Copyright (C) 2018 deeplearningrobotics.ai
cmake_minimum_required(VERSION 2.8.3)
project(tensorswarm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -O3 -W -Wall -Wextra -pedantic")

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    geometry_msgs
    message_generation
    roscpp
)

add_message_files(
    FILES
    Observation.msg
    LaserRay.msg
    LaserScan.msg
)
add_service_files(
    FILES
    AIService.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS message_runtime 
)

add_subdirectory(plugin)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
