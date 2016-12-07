# CMake generated Testfile for 
# Source directory: D:/OneDrive/Documenten/Werk/Documents/Projects/MotionCompensation/GitHubRepos/PET-Motion-Correction/STIR-master/src/test
# Build directory: D:/STIRBUILD/src/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_ROIs "D:/STIRBUILD/src/test/test_ROIs")
add_test(test_warp_image "D:/STIRBUILD/src/test/test_warp_image")
add_test(test_DynamicDiscretisedDensity "D:/STIRBUILD/src/test/test_DynamicDiscretisedDensity")
add_test(test_Array "D:/STIRBUILD/src/test/test_Array")
add_test(test_ArrayFilter "D:/STIRBUILD/src/test/test_ArrayFilter")
add_test(test_NestedIterator "D:/STIRBUILD/src/test/test_NestedIterator")
add_test(test_VectorWithOffset "D:/STIRBUILD/src/test/test_VectorWithOffset")
add_test(test_convert_array "D:/STIRBUILD/src/test/test_convert_array")
add_test(test_IndexRange "D:/STIRBUILD/src/test/test_IndexRange")
add_test(test_coordinates "D:/STIRBUILD/src/test/test_coordinates")
add_test(test_filename_functions "D:/STIRBUILD/src/test/test_filename_functions")
add_test(test_VoxelsOnCartesianGrid "D:/STIRBUILD/src/test/test_VoxelsOnCartesianGrid")
add_test(test_zoom_image "D:/STIRBUILD/src/test/test_zoom_image")
add_test(test_ByteOrder "D:/STIRBUILD/src/test/test_ByteOrder")
add_test(test_Scanner "D:/STIRBUILD/src/test/test_Scanner")
add_test(test_ArcCorrection "D:/STIRBUILD/src/test/test_ArcCorrection")
add_test(test_find_fwhm_in_image "D:/STIRBUILD/src/test/test_find_fwhm_in_image")
add_test(test_proj_data_info "D:/STIRBUILD/src/test/test_proj_data_info")
add_test(test_proj_data_in_memory "D:/STIRBUILD/src/test/test_proj_data_in_memory")
add_test(test_export_array "D:/STIRBUILD/src/test/test_export_array")
add_test(test_linear_regression "D:/STIRBUILD/src/test/test_linear_regression" "D:/OneDrive/Documenten/Werk/Documents/Projects/MotionCompensation/GitHubRepos/PET-Motion-Correction/STIR-master/src/test/input/test_linear_regression.in")
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test_stir_math "D:/STIRBUILD/src/test/Debug/test_stir_math.exe" "D:/STIRBUILD/src/utilities/Debug/stir_math.exe")
  set_tests_properties(test_stir_math PROPERTIES  DEPENDS "stir_math")
elseif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test_stir_math "D:/STIRBUILD/src/test/Release/test_stir_math.exe" "D:/STIRBUILD/src/utilities/Release/stir_math.exe")
  set_tests_properties(test_stir_math PROPERTIES  DEPENDS "stir_math")
elseif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test_stir_math "D:/STIRBUILD/src/test/MinSizeRel/test_stir_math.exe" "D:/STIRBUILD/src/utilities/MinSizeRel/stir_math.exe")
  set_tests_properties(test_stir_math PROPERTIES  DEPENDS "stir_math")
elseif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test_stir_math "D:/STIRBUILD/src/test/RelWithDebInfo/test_stir_math.exe" "D:/STIRBUILD/src/utilities/RelWithDebInfo/stir_math.exe")
  set_tests_properties(test_stir_math PROPERTIES  DEPENDS "stir_math")
else()
  add_test(test_stir_math NOT_AVAILABLE)
endif()
add_test(test_OutputFileFormat_test_InterfileOutputFileFormat.in "D:/STIRBUILD/src/test/test_OutputFileFormat" "D:/OneDrive/Documenten/Werk/Documents/Projects/MotionCompensation/GitHubRepos/PET-Motion-Correction/STIR-master/src/test/input/test_InterfileOutputFileFormat.in")
add_test(test_OutputFileFormat_test_InterfileOutputFileFormat_short.in "D:/STIRBUILD/src/test/test_OutputFileFormat" "D:/OneDrive/Documenten/Werk/Documents/Projects/MotionCompensation/GitHubRepos/PET-Motion-Correction/STIR-master/src/test/input/test_InterfileOutputFileFormat_short.in")
