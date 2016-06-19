cmake_minimum_required(VERSION 2.6)
project (adaboost_tracking)

SET(CMAKE_CXX_FLAGS "-std=c++0x")				# New C11
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)			# Binary file location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})			# Exectuable file location for main and subprojects 
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)		# Library file location for main and subprojects
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)			# Source folder
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)		# Header folder

#set(CMAKE_BUILD_TYPE debug)					# choose if you want to compile with or without debugging flags,
set(CMAKE_BUILD_TYPE release)					# uncomment only one of the following two commands at a time

#To build the project openCV is required
find_package( OpenCV REQUIRED )

#Include Required Directories
include_directories(${PROJECT_INCLUDE_DIR})			
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


 #Create a shared library (To use library creation uncomment the following)
	add_library(	
		tracking_Lib 
		${PROJECT_SOURCE_DIR}/AdaBoost.cpp
		${PROJECT_SOURCE_DIR}/WeakClassifier.cpp
		${PROJECT_SOURCE_DIR}/Types.h
		)


add_executable(nemo_tracking ${PROJECT_SOURCE_DIR}/nemo_tracking.cpp)
target_link_libraries( nemo_tracking tracking_Lib  )
target_link_libraries( nemo_tracking ${OpenCV_LIBS} )
 
	

