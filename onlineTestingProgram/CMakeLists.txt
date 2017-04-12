# Copyright: (C) 2010 RobotCub Consortium
# Author: Paul Fitzpatrick
# CopyPolicy: Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
# YARP needs CMake 2.8.9 or greater
cmake_minimum_required(VERSION 2.8.9)
# find YARP
find_package(YARP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ICUB)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${YARP_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ICUB_MODULE_PATH})

# add YARP include directories
include_directories(${ICUB_INCLUDE_DIRS} ${YARP_INCLUDE_DIRS})


# set up our programs
add_executable(controller controller.cpp)
#add_executable(network main.cpp cnn.cpp utils.cpp)
add_executable(vision vision.cpp)
add_executable(worldManipulator worldManipulator.cpp)
add_executable(fingerGrasper fingerGrasper.cpp)
add_executable(screenWriter screenWriter.cpp)

# link with YARP libraries
target_link_libraries(controller ${YARP_LIBRARIES} ${OpenCV_LIBS})
#target_link_libraries(network ${YARP_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(vision ${YARP_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(worldManipulator ${YARP_LIBRARIES})
target_link_libraries(fingerGrasper ${YARP_LIBRARIES})
target_link_libraries(screenWriter ${YARP_LIBRARIES} ${OpenCV_LIBS})









