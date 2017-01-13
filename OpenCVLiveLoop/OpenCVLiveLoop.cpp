// OpenCVLiveLoop.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "stdafx.h"
#include "MonoLoop.h"
#include "StereoLoop.h"
#include <iostream>
#include "OpenCVLiveLoop.h"
using namespace std;

const int MY_IMAGE_WIDTH = 640;
const int MY_IMAGE_HEIGHT = 480;
const int MY_WAIT_IN_MS = 20;



/**********************************************************/
cv::Mat blackImg = cv::Mat(MY_IMAGE_HEIGHT, MY_IMAGE_WIDTH, 16, cv::Scalar(0, 0, 0));
cv::Mat bgImg;
cv::Mat currFrame;
const int PREV_IMG_DIFFS_SIZE = 1;
cv::Mat prevImgDiffs[PREV_IMG_DIFFS_SIZE];


//Einzelne Positions-Variablen
int minX, minY, maxX, maxY;
int armMinX, armMinY, armMaxX, armMaxY;

//Variablen zum speichern der letzten Positions-Variablen
const int POSITION_VARS = 8;
const int POSITIONS_ACCURACY = 20;
int positions[POSITIONS_ACCURACY][POSITION_VARS];
int positionsAverageDelta[POSITION_VARS];
int positionsCount = 0;
enum PositionVars {
	MIN_X, MIN_Y, MAX_X, MAX_Y,
	ARM_MIN_X, ARM_MIN_Y, ARM_MAX_X, ARM_MAX_Y
};
enum Directions {
	NO_DIRECTION, LEFT, RIGHT
};
Directions direction = NO_DIRECTION;

//Zustände/Bedingungen, die erreicht/erfüllt werden müssen, um eine bestimmte Animation auszuführen
enum Conditions {
	NO_CONDITION,
	KAMEHAMEHA_START,
	KAMEHAMEHA_DETECTED_HAND_MOVEMENT,
	KAMEHAMÈHA_CORRECT_HAND_MOVEMENT
};
const int ENTERED_STATES_MAX = 5;
Conditions enteredStates[ENTERED_STATES_MAX];
int conditionWaitCount = 0;
string cascadeKamehamehaLeft_path = "C:/Users/Nick/Studium/Medien- und Kommunikationsinformatik/6. Semester/(W) Bildverarbeitung - Labor/Aufgabe05/OpenCV_Kamehameha/Cascade Classifier/Cascade_kamehameha_left_V2.xml";
cv::CascadeClassifier cascadeKamehamehaLeft;
vector<cv::Rect> cascadeKamehamehaRects;

//Mögliche Gesten/Animationen
enum Gestures {
	NO_GESTURE,
	KAMEHAMEHA_LEFT,
	KAMEHAMEHA_RIGHT
};
Gestures gesture = NO_GESTURE;


bool firstLoop = true;
int aniFrameNumber = 0;

int kamehamehaX = (int)(MY_IMAGE_WIDTH  / 2.0);
int kamehamehaY = (int)(MY_IMAGE_HEIGHT / 2.0);



/*******************Positions-Funktionen************************************************************************/
void clearPositions() {
	for (int i = 0; i < POSITION_VARS; i++) {
		for (int j = 0; j < POSITIONS_ACCURACY; j++) {
			positions[j][i] = 0;
		}
	}
}

void manipulatePositions(int index, PositionVars posVar, int value) {
	if (index > -1) {
		positions[index][posVar] = value;
	}
	else {
		index = POSITIONS_ACCURACY + index;
		positions[index][posVar] = value;
	}
}

bool isEmptyPositionsEntry(int index) {
	for (int i = 0; i < POSITION_VARS; i++) {
		if (positions[index][i] != 0) {
			return false;
		}
	}
	return true;
}

int getLastPosition(int returnBy, PositionVars posVar) {
	int start = positionsCount - returnBy;
	if (start > -1) {
		if (isEmptyPositionsEntry(start)) return INT16_MIN;
		return positions[start][posVar];
	}
	else {
		start = POSITIONS_ACCURACY + start;
		if (isEmptyPositionsEntry(start)) return INT16_MIN;
		return positions[start][posVar];
	}
}

int getPositionAverageDelta(int returnBy, PositionVars posVar, int skip = 0) {
	int start = positionsCount - skip;
	if (positionsCount == 0) {
		int a = 0;
	}
	int curr = 0;
	int prev = 0;
	int averageDelta = 0;
	bool jump = false;
	for (int i = start; i > (start - returnBy); i--) {
		if (i < 0) {
			jump = true;
			break;
		}
		if (i == start) {
			if (isEmptyPositionsEntry(i)) return INT16_MIN;
			prev = positions[i][posVar];
			continue;
		}
		if (isEmptyPositionsEntry(i)) return INT16_MIN;
		curr = positions[i][posVar];
		averageDelta += prev - curr;
		prev = curr;
	}
	if (jump) {
		for (int i = (POSITIONS_ACCURACY - 1); i > (POSITIONS_ACCURACY + (start - returnBy)); i--) {
			if (isEmptyPositionsEntry(i)) return INT16_MIN;
			curr = positions[i][posVar];
			averageDelta += prev - curr;
			prev = curr;
		}
	}
	averageDelta = averageDelta / ((float)returnBy - 1);

	return averageDelta;
}

int getPositionAverageDeltaBetw2Pos(int returnBy, PositionVars posVar1, PositionVars posVar2, int skip = 0) {
	int start = positionsCount - skip;
	if (positionsCount == 0) {
		int a = 0;
	}
	int curr = 0;
	int prev = 0;
	int averageDelta = 0;
	bool jump = false;
	for (int i = start; i > (start - returnBy); i--) {
		if (i < 0) {
			jump = true;
			break;
		}
		if (i == start) {
			if (isEmptyPositionsEntry(i)) return INT16_MIN;
			prev = positions[i][posVar1] - positions[i][posVar2];
			continue;
		}
		if (isEmptyPositionsEntry(i)) return INT16_MIN;
		curr = positions[i][posVar1] - positions[i][posVar2];
		averageDelta += prev - curr;
		prev = curr;
	}
	if (jump) {
		for (int i = (POSITIONS_ACCURACY - 1); i > (POSITIONS_ACCURACY + (start - returnBy)); i--) {
			if (isEmptyPositionsEntry(i)) return INT16_MIN;
			curr = positions[i][posVar1] - positions[i][posVar2];
			averageDelta += prev - curr;
			prev = curr;
		}
	}
	averageDelta = averageDelta / ((float)returnBy - 1);

	return averageDelta;
}
/**********************************************************************************************************/

void clearEnteredStates() {
	for (int i = 0; i < ENTERED_STATES_MAX; i++) {
		enteredStates[i] = NO_CONDITION;
	}
}




/*******************Animations-Funktionen*******************************************************************/
cv::Mat resizeAndPosAnimation(cv::Mat aniImg, float resizeFactor, int originX, int originY) {
	cv::Mat addImg;
	blackImg.copyTo(addImg);
	cout << "originX: " << originX << "          originY: " << originY << endl;

	//Animations-Bild auf passende Größe skalieren
	int newAniImgWidth = (int)(aniImg.cols * resizeFactor);
	int newAniImgHeight = (int)(aniImg.rows * resizeFactor);
	cv::resize(aniImg, aniImg, cv::Size(newAniImgWidth, newAniImgHeight));	

	cv::Mat aniImg_ROI;
	aniImg.copyTo(aniImg_ROI);

	int originXOffset = 0;
	int originYOffset = (int)(newAniImgHeight / 2.0);
	if (direction == RIGHT) {
		originXOffset = (int)(120 * resizeFactor);
	}
	else {
		originXOffset = (int)(520 * resizeFactor);
	}

	//Animations-Bild verschieben und clippen
	int deltaX = originX - originXOffset;
	int deltaY = originY - originYOffset;
	cout << "deltaX: " << deltaX << "          deltaY: " << deltaY << endl;
	cout << "aniImg-Width: " << aniImg.cols << "          aniImg-Height: " << aniImg.rows << endl;

	int clippingX = 0;
	int clippingY = 0;
	int clippingW = 0;
	int clippingH = 0;
	int positionX = 0;
	int positionY = 0;
	if (deltaX < 0) {
		clippingX = deltaX * (-1);
		positionX = 0;
		
		clippingW = aniImg.cols + deltaX;
		if (clippingW > addImg.cols) {
			clippingW = addImg.cols;
		}
	}
	else {
		clippingX = 0;
		positionX = deltaX;

		clippingW = aniImg.cols;
		if ((clippingW + deltaX) > addImg.cols) {
			clippingW = addImg.cols - deltaX;
		}
	}
	if (deltaY < 0) {
		clippingY = deltaY * (-1);
		positionY = 0;

		clippingH = aniImg.rows + deltaY;
		if (clippingH > addImg.rows) {
			clippingH = addImg.rows;
		}
	}
	else {
		clippingY = 0;
		positionY = deltaY;

		clippingH = aniImg.rows;
		if ((clippingH + deltaY) > addImg.rows) {
			clippingH = addImg.rows - deltaY;
		}
	}

	cout << "clippingX: " << clippingX << "          clippingY: " << clippingY << endl;
	cout << "positionX: " << positionX << "          positionY: " << positionY << endl;

	//Clippen
	aniImg_ROI(cv::Rect(clippingX, clippingY, clippingW, clippingH)).copyTo(aniImg_ROI);
	imshow("aniImgClipped", aniImg_ROI);

	//Animations-Bild an gewünschte Position setzen
	aniImg_ROI.copyTo(addImg(cv::Rect(positionX, positionY, aniImg_ROI.cols, aniImg_ROI.rows)));
	imshow("aniImg_2", addImg);

	/*
	int offsetX;
	if (direction == RIGHT) {
		offsetX = (int)(520 * resizeFactor);
	}
	else {
		offsetX = (int)(120 * resizeFactor);
	}

	//Animations-Bild auf passende Größe skalieren
	int newAniImgWidth  = (int)(aniImg.cols * resizeFactor);
	int newAniImgHeight = (int)(aniImg.rows * resizeFactor);
	cv::resize(aniImg, aniImg, cv::Size(newAniImgWidth, newAniImgHeight));
	//imshow("aniImg_resized", aniImg);

	blackImg.copyTo(addImg);

	//Animations-Bild verschieben und clippen
	cv::Mat aniImg_ROI;
	aniImg.copyTo(aniImg_ROI);

	int deltaX = addImg.cols - aniImg.cols - (addImg.cols - originX) + offsetX;
	int deltaY = addImg.rows - aniImg.rows - (addImg.rows - originY) + (int)(aniImg.rows / 2.0);

	int clippingX, clippingY, clippingW, clippingH;
	if (deltaX < 0) {
		clippingX = deltaX * (-1);
		deltaX = 0;
	}
	else clippingX = 0;
	if (deltaY < 0) {
		clippingY = deltaY * (-1);
		deltaY = 0;
	} else clippingY = 0;
	clippingW = ((aniImg.cols - clippingX) > addImg.cols) ? (addImg.cols) : (aniImg.cols - clippingX - deltaX);
	clippingH = ((aniImg.rows - clippingY) > addImg.rows) ? (addImg.rows) : (aniImg.rows - clippingY - deltaY);

	aniImg_ROI(cv::Rect(clippingX, clippingY, clippingW, clippingH)).copyTo(aniImg_ROI);
	imshow("aniImg_ROI", aniImg_ROI);

	//Animations-Bild an gewünschte Position setzen
	aniImg_ROI.copyTo(addImg(cv::Rect(abs(deltaX), abs(deltaY), aniImg_ROI.cols, aniImg_ROI.rows)));
	imshow("aniImg_2", addImg);
	*/


	return addImg;
}

string getAnimationFrameNumberAsString(int animationFrameNumber)
{
	int tempNum = animationFrameNumber;
	int digits = 0;
	if (tempNum <= 0) digits = 1;
	while (tempNum) {
		tempNum /= 10;
		digits++;
	}

	string out = "";
	while ((5 - digits) > 0) {
		out += "0";
		digits++;
	}

	out += to_string(animationFrameNumber);

	return out;
}

cv::Mat getAniImg(int animationFrameNumber) {
	cv::Mat aniImg;
	stringstream path;
	if (gesture == KAMEHAMEHA_LEFT) {
		path << "C:/Users/Nick/Studium/Medien- und Kommunikationsinformatik/6. Semester/(W) Bildverarbeitung - Labor/Aufgabe05/OpenCV_Kamehameha/Animationen/Kamehameha-LEFT-JPG-640x480/Kamehameha_" << getAnimationFrameNumberAsString(animationFrameNumber) << ".jpg";
	} else if (gesture == KAMEHAMEHA_RIGHT) {
		path << "C:/Users/Nick/Studium/Medien- und Kommunikationsinformatik/6. Semester/(W) Bildverarbeitung - Labor/Aufgabe05/OpenCV_Kamehameha/Animationen/Kamehameha-RIGHT-JPG-640x480/Kamehameha_" << getAnimationFrameNumberAsString(animationFrameNumber) << ".jpg";
	}
	
	aniImg = cv::imread(path.str(), CV_LOAD_IMAGE_COLOR);

	if (aniImg.data == NULL) {
		aniFrameNumber = 0;
		gesture = NO_GESTURE;
		direction = NO_DIRECTION;
		clearPositions();
	}
	else {
		aniFrameNumber++;
	}
	
	return aniImg;
}
/**********************************************************************************************************/



bool searchPositions(cv::Mat imgDiff) {
	float scaleFactor = 0.15625;

	cv::Mat imgDiffSmall;
	imgDiff.copyTo(imgDiffSmall);
	cv::resize(imgDiffSmall, imgDiffSmall, cv::Size((int)(imgDiffSmall.cols * scaleFactor), (int)(imgDiffSmall.rows * scaleFactor)));
	imshow("imgSmall", imgDiffSmall);

	minX = INT16_MAX;
	minY = INT16_MAX;
	maxX = -1;
	maxY = -1;
	cv::Point leftP;
	cv::Point topP;
	cv::Point rightP;
	cv::Point bottomP;
	for (int y = 0; y < imgDiffSmall.rows; y++) {
		for (int x = 0; x < imgDiffSmall.cols; x++) {
			cv::Scalar pixel = imgDiffSmall.at<uchar>(y, x);
			if (pixel.val[0] > 254) {

				if (minX > x) {
					minX = x;
					leftP = cv::Point(x, y);
				}
				else {
					minX = minX;
				}

				if (minY > y) {
					minY = y;
					topP = cv::Point(x, y);
				}
				else {
					minY = minY;
				}

				if (maxX < x) {
					maxX = x;
					rightP = cv::Point(x, y);
				}
				else {
					maxX = maxX;
				}

				if (maxY < y) {
					maxY = y;
					bottomP = cv::Point(x, y);
				}
				else {
					maxY = maxY;
				}

			}
		}
	}

	//Arm-Bereich detektieren
	int areaWidth = 150;
	int armMinX_l = minX;
	int armMinY_l = INT16_MAX;
	int armMaxX_l = maxX;
	int armMaxY_l = -1;
	if ((minX + (int)(areaWidth * scaleFactor)) < imgDiffSmall.cols && (minX + (int)(areaWidth * scaleFactor)) > 0) {
		for (int y1 = minY; y1 < maxY; y1++) {
			for (int x1 = minX; x1 < minX + (int)(areaWidth * scaleFactor); x1++) {
				cv::Scalar pixel = imgDiffSmall.at<uchar>(y1, x1);
				if (pixel.val[0] > 254) {

					if (armMinY_l > y1) {
						armMinY_l = y1;
					}
					else {
						armMinY_l = armMinY_l;
					}

					if (armMaxY_l < y1) {
						armMaxY_l = y1;
					}
					else {
						armMaxY_l = armMaxY_l;
					}

				}
			}
		}
	}
	
	cout << "maxX" << maxX - (int)(areaWidth * scaleFactor) << endl;
	int armMinX_r = minX;
	int armMinY_r = INT16_MAX;
	int armMaxX_r = maxX;
	int armMaxY_r = -1;
	if ((maxX - (int)(areaWidth * scaleFactor)) < imgDiffSmall.cols && (maxX - (int)(areaWidth * scaleFactor)) > 0) {
		for (int y2 = minY; y2 < maxY; y2++) {
			for (int x2 = maxX - (int)(areaWidth * scaleFactor); x2 < maxX; x2++) {
				cv::Scalar pixel = imgDiffSmall.at<uchar>(y2, x2);
				if (pixel.val[0] > 254) {

					if (armMinY_r > y2) {
						armMinY_r = y2;
					}
					else {
						armMinY_r = armMinY_r;
					}

					if (armMaxY_r < y2) {
						armMaxY_r = y2;
					}
					else {
						armMaxY_r = armMaxY_r;
					}

				}
			}
		}
	}

	armMinX = minX;
	armMaxX = maxX;
	if (direction == NO_DIRECTION) {
		if (armMinY_l > armMinY_r) {
			armMinY = armMinY_l;
			armMaxY = armMaxY_l;
			direction = LEFT;
		}
		else {
			armMinY = armMinY_r;
			armMaxY = armMaxY_r;
			direction = RIGHT;
		}
	}
	else if (direction == LEFT) {
		armMinY = armMinY_l;
		armMaxY = armMaxY_l;
	}
	else if (direction == RIGHT) {
		armMinY = armMinY_r;
		armMaxY = armMaxY_r;
	}
	

	/*
	int personMinX = INT16_MAX;
	int personMinY = minY;
	int personMaxX = maxX;
	int personMaxY = maxX;
	for (int y2 = minY; y2 < maxY; y2++) {
		if (y2 == armMinY) {
			y2 = armMaxY;
			continue;
		}
		for (int x2 = minX; x2 < maxX; x2++) {
			cv::Scalar pixel = imgDiffSmall.at<uchar>(y2, x2);
			if (pixel.val[0] > 254) {

				if (personMinX > x2) {
					personMinX = x2;
					break;
				}

			}
		}
	}
	armMaxX = personMinX;
	*/
	
	if (minX > -1 && minX < imgDiffSmall.cols &&
		minY > -1 && minY < imgDiffSmall.rows &&
		maxX > -1 && maxX < imgDiffSmall.cols &&
		maxY > -1 && maxY < imgDiffSmall.rows &&

		armMinX > -1 && armMinX < imgDiffSmall.cols &&
		armMinY > -1 && armMinY < imgDiffSmall.rows &&
		armMaxX > -1 && armMaxX < imgDiffSmall.cols &&
		armMaxY > -1 && armMaxY < imgDiffSmall.rows /*&&
		
		personMinX > -1 && personMinX < imgDiffSmall.cols &&
		personMinY > -1 && personMinY < imgDiffSmall.rows &&
		personMaxX > -1 && personMaxX < imgDiffSmall.cols &&
		personMaxY > -1 && personMaxY < imgDiffSmall.rows*/) {
		//Bewegung erkannt

		minX	= (int)(minX / scaleFactor);
		minY	= (int)(minY / scaleFactor);
		maxX	= (int)(maxX / scaleFactor);
		maxY	= (int)(maxY / scaleFactor);
		leftP	= cv::Point((int)(leftP.x / scaleFactor), (int)(leftP.y / scaleFactor));
		topP	= cv::Point((int)(topP.x / scaleFactor), (int)(topP.y / scaleFactor));
		rightP	= cv::Point((int)(rightP.x / scaleFactor), (int)(rightP.y / scaleFactor));
		bottomP = cv::Point((int)(bottomP.x / scaleFactor), (int)(bottomP.y / scaleFactor));

		
		armMinX = (int)(armMinX / scaleFactor);
		armMinY = (int)(armMinY / scaleFactor);
		armMaxX = (int)(armMaxX / scaleFactor);
		armMaxY = (int)(armMaxY / scaleFactor);
		
		/*
		personMinX = (int)(personMinX / scaleFactor);
		personMinY = (int)(personMinY / scaleFactor);
		personMaxX = (int)(personMaxX / scaleFactor);
		personMaxY = (int)(personMaxY / scaleFactor);
		*/

		/****************************/
		cv::Mat imgGraphics;
		currFrame.copyTo(imgGraphics);

		cv::rectangle(imgGraphics, cv::Rect(cv::Point(minX, minY), cv::Point(maxX, maxY)), cv::Scalar(0, 0, 255), 1);
		cv::circle(imgGraphics, leftP, 1, cv::Scalar(255, 0, 0), 5);
		cv::circle(imgGraphics, topP, 1, cv::Scalar(255, 0, 0), 5);		
		cv::circle(imgGraphics, rightP, 1, cv::Scalar(255, 0, 0), 5);
		cv::circle(imgGraphics, bottomP, 1, cv::Scalar(255, 0, 0), 5);

		//cv::rectangle(imgGraphics, cv::Rect(cv::Point(personMinX, personMinY), cv::Point(personMaxX, personMaxY)), cv::Scalar(0, 255, 255), 1);
		cv::rectangle(imgGraphics, cv::Rect(cv::Point(armMinX, armMinY), cv::Point(armMaxX, armMaxY)), cv::Scalar(0, 255, 0), 1);

		imshow("imgGraphics", imgGraphics);
		/****************************/
		return true;
	}
	return false;
}


void setKamehamehaXY(PositionVars posVarX, int boundary, int softness) {
	if (aniFrameNumber < 2) {
		if (posVarX == ARM_MIN_X) kamehamehaX = armMinX;
		else if (posVarX == ARM_MAX_X) kamehamehaX = armMaxX;
		
		kamehamehaY = armMinY + (int)((armMaxY - armMinY) / 2);
	}
	else {
		int xAverage = getPositionAverageDelta(2, posVarX);
		int newX = 0;
		if (abs(xAverage) > boundary) {
			if (xAverage < 0) newX = getLastPosition(1, posVarX) - softness;
			else newX = getLastPosition(1, posVarX) + softness;
			manipulatePositions(positionsCount, posVarX, newX);
		}
		else {
			if(posVarX == ARM_MIN_X) newX = armMinX;
			else if(posVarX == ARM_MAX_X) newX = armMaxX;
			
		}

		int armMinYAverage = getPositionAverageDelta(2, ARM_MIN_Y);
		int newArmMinY = 0;
		if (abs(armMinYAverage) > boundary) {
			if (armMinYAverage < 0) newArmMinY = getLastPosition(1, ARM_MIN_Y) - softness;
			else newArmMinY = getLastPosition(1, ARM_MIN_Y) + softness;
			manipulatePositions(positionsCount, ARM_MIN_Y, newArmMinY);
		}
		else {
			newArmMinY = armMinY;
		}
		int armMaxYAverage = getPositionAverageDelta(2, ARM_MAX_Y);
		int newArmMaxY = 0;
		if (abs(armMaxYAverage) > boundary) {
			if (armMaxYAverage < 0) newArmMaxY = getLastPosition(1, ARM_MAX_Y) - softness;
			else newArmMaxY = getLastPosition(1, ARM_MAX_Y) + softness;
			manipulatePositions(positionsCount, ARM_MAX_Y, newArmMaxY);
		}
		else {
			newArmMaxY = armMaxY;
		}

		kamehamehaX = newX;
		kamehamehaY = newArmMinY + (int)((newArmMaxY - newArmMinY) / 2);
	}
}


void searchGesture(cv::Mat frame) {
	cv::Mat img;
	frame.copyTo(img);

	cv::Mat imgDiff;
	cv::absdiff(bgImg, img, imgDiff);
	cv::cvtColor(imgDiff, imgDiff, CV_BGR2GRAY);
	cv::blur(imgDiff, imgDiff, cv::Size(12, 12));
	cv::threshold(imgDiff, imgDiff, 20, 255, CV_THRESH_BINARY);/**/

	cv::Mat tempDiff;
	imgDiff.copyTo(tempDiff);
	for (int i = 0; i < PREV_IMG_DIFFS_SIZE; i++) {
		if (prevImgDiffs[0].data != NULL) {
			cv::bitwise_or(imgDiff, prevImgDiffs[0], imgDiff);
		}
	}
	for (int i = 0; i < PREV_IMG_DIFFS_SIZE; i++) {
		if (i == PREV_IMG_DIFFS_SIZE - 1) {
			tempDiff.copyTo(prevImgDiffs[i]);
		}
		else {
			prevImgDiffs[i + 1].copyTo(prevImgDiffs[i]);
		}		
	}
	
	
	imshow("diffImg", imgDiff);

	if (!searchPositions(imgDiff)) {
		return;
	}
	
	int currentPositions[POSITION_VARS] = { minX, minY, maxX, maxY, armMinX, armMinY, armMaxX, armMaxY };
	if (positionsCount >= POSITIONS_ACCURACY) {
		positionsCount = 0;
	}
	for (int i = 0; i < POSITION_VARS; i++) {
		positions[positionsCount][i] = currentPositions[i];
	}

	
	//enteredStates[0] = KAMEHAMEHA_START;
	//enteredStates[1] = KAMEHAMEHA_DETECTED_HAND_MOVEMENT;
	//Bedingungen für Kamehameha überprüfen
	/*
	if (gesture != KAMEHAMEHA) {
		if (enteredStates[2] == KAMEHAMÈHA_CORRECT_HAND_MOVEMENT) {
			gesture = KAMEHAMEHA;
			clearEnteredStates();
		}
		else {
			if (enteredStates[1] == KAMEHAMEHA_DETECTED_HAND_MOVEMENT) {
				
				cv::Mat greyImg;
				img.copyTo(greyImg);

				cv::cvtColor(greyImg, greyImg, CV_BGR2GRAY);
				cv::equalizeHist(greyImg, greyImg);

				cascadeKamehamehaLeft.detectMultiScale(greyImg, cascadeKamehamehaRects, 1.1, 3, 0, cv::Size(30, 30));
				for (size_t i = 0; i < cascadeKamehamehaRects.size(); i++)
				{
					cv::Point center(cascadeKamehamehaRects[i].x + cascadeKamehamehaRects[i].width*0.5, cascadeKamehamehaRects[i].y + cascadeKamehamehaRects[i].height*0.5);
					ellipse(img, center, cv::Size(cascadeKamehamehaRects[i].width*0.5, cascadeKamehamehaRects[i].height*0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);
				}
				imshow("Grey", greyImg);
				imshow("Detect", img);
				

				if (conditionWaitCount < 2) {
					int minXAverage2 = getPositionAverageDelta(2, ARM_MIN_X);
					int minXAverage3 = getPositionAverageDelta(4, ARM_MIN_X);
					if (minXAverage3 > -50 && minXAverage3 < -2 && cascadeKamehamehaRects.size() > 0) {
						enteredStates[2] = KAMEHAMÈHA_CORRECT_HAND_MOVEMENT;
						conditionWaitCount = 0;
					}
					else {
						clearEnteredStates();
					}
				}
				else {
					conditionWaitCount--;
				}
			}
			else {
				if (enteredStates[0] == KAMEHAMEHA_START) {
					int test = sizeof(positions);
					int armMinYAverageDelta = getPositionAverageDelta(2, ARM_MIN_Y);
					int minYAverageDelta = getPositionAverageDelta(2, MIN_Y);
					int prevMinYAverageDelta = getPositionAverageDelta(3, MIN_Y, 1);
					int prevArmMinYAverageDelta = getPositionAverageDelta(3, ARM_MIN_Y, 1);
					if (armMinYAverageDelta != INT16_MIN && minYAverageDelta != INT16_MIN &&
						prevMinYAverageDelta != INT16_MIN && prevArmMinYAverageDelta != INT16_MIN &&
						prevMinYAverageDelta == prevArmMinYAverageDelta &&
						minY != armMinY &&
						armMinYAverageDelta > 40 && abs(minYAverageDelta) < 20) {
						enteredStates[1] = KAMEHAMEHA_DETECTED_HAND_MOVEMENT;
						conditionWaitCount = 5;
					}
					else if (minY != armMinY) {
						clearEnteredStates();
					}
				}
				else {
					int minYAverageDelta = getPositionAverageDelta(3, MIN_Y);
					int armMinYAverageDelta = getPositionAverageDelta(3, ARM_MIN_Y);
					if (minYAverageDelta != INT16_MIN && armMinYAverageDelta != INT16_MIN &&
						minYAverageDelta == armMinYAverageDelta) {
						enteredStates[0] = KAMEHAMEHA_START;
					}
				}
			}
		}
	}	
	*/
	
	if (gesture != KAMEHAMEHA_LEFT && gesture != KAMEHAMEHA_RIGHT) {	
		int minY_armMinY_AvDelta = getPositionAverageDeltaBetw2Pos(2, ARM_MIN_Y, MIN_Y, 1);		
		int speedMinX = getPositionAverageDelta(3, ARM_MIN_X, 1);
		int speedMaxX = getPositionAverageDelta(3, ARM_MAX_X, 1);
		int armMinXStop = getPositionAverageDelta(2, ARM_MIN_X);
		int armMaxXStop = getPositionAverageDelta(2, ARM_MAX_X);

		int space = img.rows - (getLastPosition(1, MAX_Y) - getLastPosition(1, MIN_Y));

		if
		(
			direction == LEFT &&
			minY_armMinY_AvDelta != INT16_MIN && speedMinX != INT16_MIN && armMinXStop != INT16_MIN && space != INT16_MIN &&
			minY_armMinY_AvDelta > 20 && space < 60 &&
			(armMinXStop < 5 && speedMinX > -150 && speedMinX < -50)
		)
		{
			gesture = KAMEHAMEHA_LEFT;
		}
		else if
		(
			direction == RIGHT &&
			minY_armMinY_AvDelta != INT16_MIN && speedMaxX != INT16_MIN && armMaxXStop != INT16_MIN && space != INT16_MIN &&
			minY_armMinY_AvDelta > 20 && space < 60 &&
			(armMaxXStop > -5 && speedMaxX < 150 && speedMaxX > 50)
		)
		{
			gesture = KAMEHAMEHA_RIGHT;
		}
		else
		{
			gesture = NO_GESTURE;
			direction = NO_DIRECTION;
		}
	}

	
	if (gesture == KAMEHAMEHA_LEFT) {
		setKamehamehaXY(ARM_MIN_X, 20, 5);
	}
	else if (gesture == KAMEHAMEHA_RIGHT) {
		setKamehamehaXY(ARM_MAX_X, 20, 5);
	}

	
	
	
	

	
	/**/
	positionsCount++;
}

/**********************************************************/


int MonoLoop()
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	// Set cameras to 15fps (if wanted!!!)
	cap.set(CV_CAP_PROP_FPS, 15);

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	// Set image size
	cap.set(CV_CAP_PROP_FRAME_WIDTH, MY_IMAGE_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, MY_IMAGE_HEIGHT);

	// display the frame size that OpenCV has picked in order to check 
	cout << "cam Frame size: " << dWidth << " x " << dHeight << endl;
	cv::namedWindow("cam", CV_WINDOW_AUTOSIZE);

	cv::Mat inputFrame;
	cv::Mat outputFrame;

	while (1)
	{

		bool bSuccess = cap.read(inputFrame);

		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		/*******************************todo*****************************/
		inputFrame.copyTo(currFrame);
		inputFrame.copyTo(outputFrame);

		if (firstLoop) {
			cap.read(bgImg);
			firstLoop = false;
		}
		
		searchGesture(inputFrame);

		if (gesture != NO_GESTURE) {
			cv::Mat addImg = getAniImg(aniFrameNumber);
			cout << "KamehamehaX: " << kamehamehaX << "          KamehamehaY: " << kamehamehaY << endl;
			if (kamehamehaX > -1 && kamehamehaX < addImg.cols && kamehamehaY > -1 && kamehamehaY < addImg.rows) {
				addImg = resizeAndPosAnimation(addImg, 2, kamehamehaX, kamehamehaY);
				cv::add(inputFrame, addImg, outputFrame);
			}
		}

		inputFrame.copyTo(bgImg);
		/***************************end todo*****************************/

		imshow("cam", outputFrame);
		imshow("bgImg", bgImg);

		if (cv::waitKey(MY_WAIT_IN_MS) == 27)
		{
			cout << "ESC key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}



int _tmain(int argc, _TCHAR* argv[])
{
	//CMonoLoop myLoop;
	//  CStereoLoop myLoop;
	// myLoop.Run();

	if(!cascadeKamehamehaLeft.load(cascadeKamehamehaLeft_path)) {
		printf("--(!)Error loading\n"); return -1; 
	};

	return MonoLoop();
}
