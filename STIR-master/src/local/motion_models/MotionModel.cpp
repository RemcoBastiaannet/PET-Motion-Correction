#include "stir\MotionModel.h"



START_NAMESPACE_STIR
const char * const
LinearMotionModel::registered_name = "LinearMotionModel";

//Basic Constructor
LinearMotionModel::LinearMotionModel()
	{
		BasicCoordinate<3, float> EmptyCoord;
		EmptyCoord.fill(0.0);

		this->Offset = EmptyCoord;
		this->Slope = EmptyCoord;
		this->Time = 0.0;
	}

//General Setters
	void LinearMotionModel::setOffset(const BasicCoordinate<3, float> Offset)
	{
		this->Offset = Offset;
	}

	void LinearMotionModel::setSlope(const BasicCoordinate<3, float> Slope)
	{
		this->Slope = Slope;
	}

	void LinearMotionModel::setTime(float Time)
	{
		this->Time = Time;
	}

// General Getters
	BasicCoordinate<3, float> LinearMotionModel::getOffset()
	{
		return this->Offset;
	}

	BasicCoordinate<3, float> LinearMotionModel::getSlope()
	{
		return this->Slope;
	}


	LinearMotionModel::~LinearMotionModel()
	{
	}

	float LinearMotionModel::getTime()
	{
		return this->Time;
	}

// De eigenlijke mapping
	BasicCoordinate<3, int> LinearMotionModel::translateMappingForward(BasicCoordinate<3, int> InputCoord)
	{
		BasicCoordinate<3, int> OutputCoord;
		//Dit is gewoon een lineare mapping, dus: y = a*x+b
		for (int ix = InputCoord.get_min_index(); ix < InputCoord.get_max_index(); ix++)
			OutputCoord[ix] = (int)std::round(this->Slope[ix] * (float)InputCoord[ix] + (float)this->Offset[ix]);

		return OutputCoord;
	}

	BasicCoordinate<3, int> LinearMotionModel::translateMappingBackward(BasicCoordinate<3, int> InputCoord)
	{
		BasicCoordinate<3, int> OutputCoord;
		//Dit is gewoon een lineare mapping, dus: y = a*x+b
		for (int ix = InputCoord.get_min_index(); ix < InputCoord.get_max_index(); ix++)
			OutputCoord[ix] = (int)std::round(this->Slope[ix] * (float)InputCoord[ix] + (float)this->Offset[ix]);

		return OutputCoord;
	}
//voor nu doet deze classe niets anders dan coordinaten vertalen, op basis van tijd 

END_NAMESPACE_STIR