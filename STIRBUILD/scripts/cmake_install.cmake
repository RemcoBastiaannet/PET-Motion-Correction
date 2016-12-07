# Install script for directory: D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts

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

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/stir_subtract"
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/stir_divide"
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/count"
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/stir_print_voxel_sizes.sh"
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/estimate_scatter.sh"
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/zoom_att_image.sh"
    "D:/Manon/GitHub/PET-Motion-Correction/STIR-master/scripts/get_num_voxels.sh"
    )
endif()

