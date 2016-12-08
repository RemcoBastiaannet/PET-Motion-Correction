# Install script for directory: D:/CommitMap/STIR-master/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/STIR")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/CommitMap/STIR-BUILD/src/buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/numerics_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/data_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/display/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/recon_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/modelling_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/listmode_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/IO/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/spatial_transformation_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/Shape_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/eval_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/scatter_buildblock/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/utilities/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/scatter_utilities/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/modelling_utilities/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/listmode_utilities/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/analytic/FBP2D/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/analytic/FBP3DRP/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/iterative/OSMAPOSL/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/iterative/OSSPS/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/iterative/POSMAPOSL/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/iterative/POSSPS/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/SimSET/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/SimSET/scripts/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/recon_test/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/test/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/test/numerics/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/test/modelling/cmake_install.cmake")
  include("D:/CommitMap/STIR-BUILD/src/swig/cmake_install.cmake")

endif()

