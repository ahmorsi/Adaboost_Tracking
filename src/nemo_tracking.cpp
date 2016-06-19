/*
 * nr3.cc
 *
 *  Created on: Apr 28, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include "Types.h"
#include "AdaBoost.h"

#define _objectWindow_width 121
#define _objectWindow_height 61

#define _searchWindow_width 61
#define _searchWindow_height 61

// use 30/15 for overlapping negative examples and 120/60 for non-overlapping negative examples
#define _displacement_x 30 //120
#define _displacement_y 15 //60

void drawTrackedFrame(cv::Mat& image, cv::Point& position);
// helper function
void loadImage(const std::string& imageFile, cv::Mat& image) {
	image = cv::imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data ) {
		std::cout <<  "Could not open or find the image" << std::endl ;
		exit(1);
	}
}

void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,
		std::vector<cv::Point>& referencePoints) {
	imageSequence.clear();
	referencePoints.clear();
	std::ifstream f(trainDataFile);
	std::string line;
	while (getline(f, line)) {
		std::stringstream s(line);
		std::string imageFile;
		s >> imageFile;
		referencePoints.push_back(cv::Point());
		s >> referencePoints.back().x;
		s >> referencePoints.back().y;
		imageSequence.push_back(cv::Mat());
		loadImage(imageFile, imageSequence.back());
	}
	f.close();
}

void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint) {
	imageSequence.clear();
	std::ifstream f(testDataFile);
	std::string line;
	getline(f, line);
	std::stringstream s(line);
	s >> startingPoint.x;
	s >> startingPoint.y;
	while (getline(f, line)) {
		imageSequence.push_back(cv::Mat());
		loadImage(line, imageSequence.back());
	}
	f.close();
}

void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram) {
	histogram.clear();
	histogram.resize(256, 0);
	// compute histogram over object window at point p
	
	int startY = p.y - (_objectWindow_height/2);
	int startX = p.x - (_objectWindow_width/2);
	int endY = startY + _objectWindow_height;
	int endX = startX + _objectWindow_width;
	for(int y=startY;y<endY;++y)
		for(int x= startX;x<endX;++x){
			 if(y <0 || y >= image.rows || x < 0 || x >= image.cols)
			 	continue;
			int idx = (int)image.at<uchar>(y,x);	 
			 histogram[idx] += 1;
		}
}
void generateTrainingData(std::vector<Example>& data,const std::vector<cv::Mat>& imageSequence,const std::vector<cv::Point>& referencePoints) {

	data.clear();

	// generate positive examples ... compute histogram around each training point
	Example ex;
	for(int i=0;i<imageSequence.size();++i){
		computeHistogram(imageSequence[i],referencePoints[i],ex.attributes);
		ex.label = 1;
		data.push_back(ex);
	}
	// generate negative examples (four for each frame, at top-left, top-right, bottom-left, and bottom-right of reference window)
	// windows of negative examples can overlap with the reference window
	cv::Point top_left,top_right,bottom_left,bottom_right;
	top_left.y =   -1* _objectWindow_height/2;top_left.x =   -1* _objectWindow_width/2;
	top_right.y =   -1* _objectWindow_height/2;top_right.x =   _objectWindow_width/2;
	bottom_left.y =   _objectWindow_height/2;bottom_left.x =   -1* _objectWindow_width/2;
	bottom_right.y =    _objectWindow_height/2;bottom_right.x =   _objectWindow_width/2;
	for(int i=0;i<imageSequence.size();++i){
		ex.label = 0;
		computeHistogram(imageSequence[i],referencePoints[i]+top_left,ex.attributes);
		data.push_back(ex);
		computeHistogram(imageSequence[i],referencePoints[i]+top_right,ex.attributes);
		data.push_back(ex);
		computeHistogram(imageSequence[i],referencePoints[i]+ bottom_left,ex.attributes);
		data.push_back(ex);
		computeHistogram(imageSequence[i],referencePoints[i]+bottom_right,ex.attributes);
		data.push_back(ex);
	}
}
void findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost) {
	f32 maxConf = 0,conf;
	cv::Point bestMatch;

	// search in a window around last position

	int startY = lastPosition.y - (_searchWindow_height/2);
	int startX = lastPosition.x - (_searchWindow_width/2);
	int endY = startY + _searchWindow_height;
	int endX = startX+ _searchWindow_width;
	Vector hist;
	for(int y = startY ; y < endY ; ++y)
		for(int x = startX;x < endX;++x){
			if(y <0 || y >= image.rows || x < 0 || x >= image.cols)
			 	continue;
			computeHistogram(image,cv::Point(x,y),hist);
			conf = adaBoost.confidence(hist,1);
			if(conf > maxConf){
				maxConf = conf;
				bestMatch.y = y;
				bestMatch.x = x;
			}
					 
		}
		std::cout<<"confidence: "<<maxConf<<" at x= "<<bestMatch.x<<", y = "<<bestMatch.y<<std::endl;
	lastPosition = bestMatch;
}

int main( int argc, char** argv )
{
	if(argc != 4) {
		std::cout <<" Usage: " << argv[0] << " <training-file> <test-file> <# iterations for AdaBoost>" << std::endl;
		return -1;
	}

	u32 adaBoostIterations = atoi(argv[3]);

	// load the training frames
	std::vector<cv::Mat> imageSequence;
	std::vector<cv::Point> referencePoints;
	loadTrainFrames(argv[1], imageSequence, referencePoints);

	// generate gray-scale histograms from the training frames:
	// one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
	// four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
	std::vector<Example> trainingData;
	generateTrainingData(trainingData, imageSequence, referencePoints);
	// initialize AdaBoost and train a cascade with the extracted training data
	AdaBoost adaBoost(adaBoostIterations);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	// log error rate on training set
	u32 nClassificationErrors = 0;

	for(int i=0;i<trainingData.size();++i){
		if(adaBoost.classify(trainingData[i].attributes) != trainingData[i].label)
			 ++ nClassificationErrors;
	}
	std::cout << "Error rate on training set: " << (f32)nClassificationErrors / (f32)trainingData.size() << std::endl;

	// load the test frames and the starting position for tracking
	std::vector<Example> testImages;
	cv::Point lastPosition;
	loadTestFrames(argv[2], imageSequence, lastPosition);

	// for each frame...
	for (u32 i = 0; i < imageSequence.size(); i++) {
		// ... find the best match in a window of size
		// _searchWindow_width x _searchWindow_height around the last tracked position
		findBestMatch(imageSequence.at(i), lastPosition, adaBoost);
		// draw the result
		drawTrackedFrame(imageSequence.at(i), lastPosition);
		cv::waitKey(0);
	}
	return 0;
}
void drawTrackedFrame(cv::Mat& image, cv::Point& position) {

	cv::rectangle( image, cvPoint(position.x - (_objectWindow_width/2), position.y- (_objectWindow_height/2)),
                       cvPoint(position.x + (_objectWindow_width/2), position.y + (_objectWindow_height/2)),
                       cv::Scalar(0,0,255));
	imshow( "Nemo", image );				   
}
