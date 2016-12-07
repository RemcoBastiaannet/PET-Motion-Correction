# Install script for directory: D:/OneDrive/Documenten/Werk/Documents/Projects/MotionCompensation/GitHubRepos/PET-Motion-Correction/STIR-master/src

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
  include("D:/STIRBUILD/src/buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/numerics_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/data_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/display/cmake_install.cmake")
  include("D:/STIRBUILD/src/recon_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/modelling_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/listmode_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/IO/cmake_install.cmake")
  include("D:/STIRBUILD/src/spatial_transformation_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/Shape_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/eval_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/scatter_buildblock/cmake_install.cmake")
  include("D:/STIRBUILD/src/utilities/cmake_install.cmake")
  include("D:/STIRBUILD/src/scatter_utilities/cmake_install.cmake")
  include("D:/STIRBUILD/src/modelling_utilities/cmake_install.cmake")
  include("D:/STIRBUILD/src/listmode_utilities/cmake_install.cmake")
  include("D:/STIRBUILD/src/analytic/FBP2D/cmake_install.cmake")
  include("D:/STIRBUILD/src/analytic/FBP3DRP/cmake_install.cmake")
  include("D:/STIRBUILD/src/iterative/OSMAPOSL/cmake_install.cmake")
  include("D:/STIRBUILD/src/iterative/OSSPS/cmake_install.cmake")
  include("D:/STIRBUILD/src/iterative/POSMAPOSL/cmake_install.cmake")
  include("D:/STIRBUILD/src/iterative/POSSPS/cmake_install.cmake")
  include("D:/STIRBUILD/src/SimSET/cmake_install.cmake")
  include("D:/STIRBUILD/src/SimSET/scripts/cmake_install.cmake")
  include("D:/STIRBUILD/src/recon_test/cmake_install.cmake")
  include("D:/STIRBUILD/src/test/cmake_install.cmake")
  include("D:/STIRBUILD/src/test/numerics/cmake_install.cmake")
  include("D:/STIRBUILD/src/test/modelling/cmake_install.cmake")
  include("D:/STIRBUILD/src/swig/cmake_install.cmake")

endif()

