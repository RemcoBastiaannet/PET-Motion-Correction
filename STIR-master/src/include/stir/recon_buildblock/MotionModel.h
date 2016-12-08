#pragma once
#include <vector>
#include "../Succeeded.h"

#include "stir/common.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir\BasicCoordinate.h"

START_NAMESPACE_STIR

class MotionModel
{
public:
	MotionModel();

	MotionModel(int, float, float);

	//Setters
	stir::Succeeded setSignal(std::vector<std::vector<float>>);

	stir::Succeeded setTime(int);
	stir::Succeeded setOffset(float);
	stir::Succeeded setSlope(float);

	//Getters
	std::vector<std::vector<float>> getSignal();

	int getTime();
	float getOffset();
	float getSlope();

	//Translate coords
	BasicCoordinate<3, int> translateCoordsForward(BasicCoordinate<3, int>);
	BasicCoordinate<3, int> translateCoordsBackward(BasicCoordinate<3, int>);


private:
	bool init = false;
	std::vector<std::vector<float>> Signal;

	float Offset;
	float Slope;
	int Time;
};

END_NAMESPACE_STIR