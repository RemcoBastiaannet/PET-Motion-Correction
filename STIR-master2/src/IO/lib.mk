dir := IO

$(dir)_LIB_SOURCES := \
  OutputFileFormat.cxx \
  OutputFileFormat_default.cxx \
  InterfileOutputFileFormat.cxx \
  interfile.cxx InterfileHeader.cxx InterfilePDFSHeaderSPECT.cxx\
  InputFileFormatRegistry.cxx \
  InterfileDynamicDiscretisedDensityOutputFileFormat.cxx \
  InterfileParametricDensityOutputFileFormat.cxx \
  GIPL_ImageFormat.cxx

ifeq ($(HAVE_ITK),1)
  ITKImageInputFileFormat.cxx \
  ITKImageOutputFileFormat.cxx 
endif

ifeq ($(HAVE_LLN_MATRIX),1)
  $(dir)_LIB_SOURCES += ECAT7OutputFileFormat.cxx stir_ecat7.cxx \
	  ECAT6OutputFileFormat.cxx \
	  stir_ecat_common.cxx \
	  ECAT7ParametricDensityOutputFileFormat.cxx \
	  ECAT7DynamicDiscretisedDensityOutputFileFormat.cxx \
	  ECAT7DynamicDiscretisedDensityInputFileFormat.cxx \
	  stir_ecat6.cxx ecat6_utils.cxx
endif

ifeq ($(HAVE_AVW),1)
  $(dir)_LIB_SOURCES += stir_AVW.cxx
endif

$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk
