#pragma once

#include "stir/RegisteredObject.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

class MotionModel : public RegisteredObject<MotionModel>
{
public:
	//! Name which will be used when parsing a BackProjectorByBin object
	static const char * const registered_name;
};


class LinearMotionModel : public RegisteredParsingObject<LinearMotionModel, MotionModel> 
{
public:
	//! Name which will be used when parsing a LinearMotionModel object
	static char const * const registered_name;

	LinearMotionModel();
	~LinearMotionModel();

	//Define basic getters
	float getTime();
	BasicCoordinate<3, float> getSlope();
	BasicCoordinate<3, float> getOffset();

	//Define basic setters
	void setTime(float);
	void setSlope(BasicCoordinate<3, float>);
	void setOffset(BasicCoordinate<3, float>);

	//Do the actual translation on that timepoint
	BasicCoordinate<3, int> translateMappingForward(BasicCoordinate<3, int>);
	BasicCoordinate<3, int> translateMappingBackward(BasicCoordinate<3, int>);

private:
	BasicCoordinate<3, float> Offset;
	BasicCoordinate<3, float> Slope;
	float Time;
};

END_NAMESPACE_STIR