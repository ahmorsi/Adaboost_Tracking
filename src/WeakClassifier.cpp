/*
 * NearestMeanClassifier.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "WeakClassifier.h"
#include <cmath>
#include <iostream>
#include <algorithm> 

/*
 * Stump
 */

Stump::Stump() :
		dimension_(0),
		splitAttribute_(0),
		splitValue_(0),
		classLabelLeft_(1),
		classLabelRight_(0)
{}

void Stump::initialize(u32 dimension) {
	dimension_ = dimension;
}


f32 Stump::weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32 splitValue, u32& resultingLeftLabel) {
	f32 weight = 0;
	// accumulate the class weights for both leaves according to the split induced by splitValue on splitAttribute
	for(int i=0;i<data.size();++i)
		if(data[i].attributes[splitAttribute] < splitValue){
			if(data[i].label != classLabelLeft_)
				weight += weights[i];
		}
		else
			{
				if(data[i].label != classLabelRight_)
					weight += weights[i];
			}
	return weight; 
}

void Stump::train(const std::vector<Example>& data, const Vector& weights) {

	// find best split attribute and value
	u32 resultingLeftLabel;
	f32 w,min_w = 1000000;
	for(int a=0;a<dimension_;++a){
		// f32 splitMeanValue = 0;
		f32 minV = 100000 ,maxV =  - 100000;
		for(int i=0;i<data.size();++i){
			minV = std::min(minV,data[i].attributes[a]);
			maxV = std::max(maxV,data[i].attributes[a]);
		}
		// 	splitMeanValue += data[i].attributes[a];
		// splitMeanValue /= data.size();
		for(f32 th  = minV ;th <= maxV;th += 0.1){
		w = weightedGain(data,weights,a,th,resultingLeftLabel);
		
		if(w < min_w)
		{
			min_w = w;
			splitAttribute_ = a;
			//splitValue_ = splitMeanValue;
			splitValue_ = th;
		}
		}
	}
}

u32 Stump::classify(const Vector& v) {
	u32 label = 0;

	if(v[splitAttribute_] < splitValue_)
		label = classLabelLeft_;
	else
		label = classLabelRight_;	

	return label;
}

void Stump::classify(const std::vector<Example>& data, std::vector<u32>& classAssignments) {
	classAssignments.resize(data.size());
	for (u32 i = 0; i < data.size(); i++) {
		classAssignments.at(i) = classify(data.at(i).attributes);
	}
}
