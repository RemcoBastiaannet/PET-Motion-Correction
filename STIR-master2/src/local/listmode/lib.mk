#
#

dir := local/listmode
$(dir)_LIB_SOURCES:= \
	LmToProjDataWithMC.cxx \
	LmToProjDataWithRandomRejection.cxx

#$(dir)_REGISTRY_SOURCES:= local_motion_registries.cxx

include $(WORKSPACE)/lib.mk
