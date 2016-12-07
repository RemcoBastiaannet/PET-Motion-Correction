set(dir local_motion_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	MotionModel.cpp
    MotionModel_registries.cxx
)

include(stir_lib_target)


