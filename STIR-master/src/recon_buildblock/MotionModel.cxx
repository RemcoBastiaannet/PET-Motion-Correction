#include <vector>
#include "stir/Succeeded.h"

#include "stir/common.h"

#include "stir/recon_buildblock/MotionModel.h"
#include <iostream>

START_NAMESPACE_STIR
MotionModel::MotionModel() {
	int nTimeFrames = 10;
	float Slope = 0;
	float Offset = 0;

	this->init = true;

	this->Offset = Offset;
	this->Slope = Slope;

	std::vector<std::vector<float>> Signal(2, std::vector<float>(nTimeFrames));

	//First hack-in the time vector
	for (int iTF = 0; iTF < nTimeFrames; iTF++) {
		Signal[0][iTF] = (float)iTF;
		Signal[1][iTF] = Signal[0][iTF] * this->Slope + this->Offset;
	}

	MotionModel::setSignal(Signal);
};

MotionModel::MotionModel(int nTimeFrames, float Slope, float Offset) {
	
	this->init = true;

	this->Offset = Offset;
	this->Slope = Slope;

	std::vector<std::vector<float>> Signal(2, std::vector<float>(nTimeFrames));

	//First hack-in the time vector
	for (int iTF = 0; iTF < nTimeFrames; iTF++) {
		Signal[0][iTF] = (float)iTF;
		Signal[1][iTF] = Signal[0][iTF] * this->Slope + this->Offset;
	}

	MotionModel::setSignal(Signal);

}

stir::Succeeded MotionModel::setSignal(std::vector<std::vector<float>> Signal) {
	try{
		this->Signal = Signal;
		return stir::Succeeded::yes;
	}
	catch (...) {
		return stir::Succeeded::no;
	}
	
}


std::vector<std::vector<float>> MotionModel::getSignal() {
	return this->Signal;
}


int MotionModel::getTime() {
	return this->Time;
}

float MotionModel::getOffset() {
	return this->Offset;
}

float MotionModel::getSlope() {
	return this->Slope;
}

stir::Succeeded MotionModel::setTime(int Time) {
	this->Time = Time;
	return stir::Succeeded::yes;
}

stir::Succeeded MotionModel::setOffset(float Offset) {
	this->Offset = Offset;
	return stir::Succeeded::yes;
}

stir::Succeeded MotionModel::setSlope(float Slope) {
	this->Slope = Slope;
	return stir::Succeeded::yes;
}

BasicCoordinate<3,int> MotionModel::translateCoordsBackward(BasicCoordinate<3, int> coordsIn) {
	
	BasicCoordinate<3, int> CoordsOut;
	int numDim = coordsIn.size();
	for (int iDim = 1; iDim <= numDim; iDim++) {
		CoordsOut[iDim] = coordsIn[iDim];// *this->Slope + this->Offset;
	}
	CoordsOut[2] = CoordsOut[2] + (int)this->Offset;

	return CoordsOut;
}

BasicCoordinate<3, int> MotionModel::translateCoordsForward(BasicCoordinate<3, int> coordsIn) {

	BasicCoordinate<3, int> CoordsOut;
	int numDim = coordsIn.size();
	for (int iDim = 1; iDim <= numDim; iDim++) {
		CoordsOut[iDim] = coordsIn[iDim];// *this->Slope + this->Offset;
	}
	CoordsOut[2] = CoordsOut[2] + (int)this->Offset;

	return CoordsOut;
}

END_NAMESPACE_STIR
