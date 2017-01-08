// OpenCVLiveLoop.cpp : Definiert den Einstiegspunkt f�r die Konsolenanwendung.
//

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

vector<vector<cv::Point>> contours;
vector<cv::Vec4i> hierarchy;

int minX, minY, maxX, maxY;
int armMinX, armMinY, armMaxX, armMaxY;
const int POSITION_VARS = 8;
const int POSITIONS_ACCURACY = 20;
const int AVERAGE = 5;
int positions[POSITIONS_ACCURACY][POSITION_VARS];
int positionsCount = 0;

bool firstLoop = true;
bool aniFinished = true;
int aniFrameNumber = 0;

enum Gestures {
	NONE,
	KAMEHAMEHA
};
Gestures gesture = NONE;
enum movementDirections {
	LEFT, RIGHT
};
movementDirections movementDirection;
const int MOVEMENT_ACCURACY = 100;
int movementCounter = 0;
int minXPosistions[MOVEMENT_ACCURACY];
int maxXPosistions[MOVEMENT_ACCURACY];

int kamehamehaX = (int)(MY_IMAGE_WIDTH  / 2.0);
int kamehamehaY = (int)(MY_IMAGE_HEIGHT / 2.0);

void clearMovements() {
	for (int i = 0; i < MOVEMENT_ACCURACY; i++) {
		minXPosistions[i] = -1;
		maxXPosistions[i] = -1;
	}
	movementCounter = 0;
}

void clearPositions() {
	for (int i = 0; i < POSITION_VARS; i++) {
		for (int j = 0; j < POSITIONS_ACCURACY; j++) {
			positions[j][i] = 0;
		}
	}
}

cv::Mat resizeAndPosAnimation(cv::Mat aniImg, float resizeFactor, int originX, int originY) {
	cv::Mat addImg;
	int offsetX = (int)(120 * resizeFactor);

	//Animations-Bild auf passende Gr��e skalieren
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
	//imshow("aniImg_ROI", aniImg_ROI);

	//Animations-Bild an gew�nschte Position setzen
	aniImg_ROI.copyTo(addImg(cv::Rect(abs(deltaX), abs(deltaY), aniImg_ROI.cols, aniImg_ROI.rows)));
	imshow("aniImg_2", addImg);

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
	path << "C:/Users/Nick/Studium/Medien- und Kommunikationsinformatik/6. Semester/(W) Bildverarbeitung - Labor/Aufgabe05/Animationen/Kamehameha-JPG-640x480/Kamehameha_" << getAnimationFrameNumberAsString(animationFrameNumber) << ".jpg";
	aniImg = cv::imread(path.str(), CV_LOAD_IMAGE_COLOR);

	if (aniImg.data == NULL) {
		aniFrameNumber = 0;
		gesture = NONE;
		clearPositions();
	}
	else {
		aniFrameNumber++;
	}
	
	return aniImg;
}

Gestures checkMovement() {
	/*
	if (minXPosistions[MOVEMENT_ACCURACY] < 0) {
		return false;
	}
	*/
	int minX_mediumDev = 0;
	int realMovementLength = 1;
	for (int i = 1; i < MOVEMENT_ACCURACY; i++) {
		if (minXPosistions[i] <= 0) {
			if (i == 1) {
				return NONE;
			}
			else {
				realMovementLength = i;
				break;
			};
		}
		minX_mediumDev += minXPosistions[i - 1] - minXPosistions[i];
	}
	minX_mediumDev = (int)((double)minX_mediumDev / (double)realMovementLength);
	if (minX_mediumDev < 0) {
		movementDirection = RIGHT;
	}
	else {
		movementDirection = LEFT;
	}

	if (minX_mediumDev > 30) {
		return KAMEHAMEHA;
	}


	return NONE;
}

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
	armMinX = minX;
	armMinY = INT16_MAX;
	armMaxX = maxX;
	armMaxY = -1;
	if ((minX + (int)(150 * scaleFactor)) < imgDiffSmall.cols) {
		for (int y1 = minY; y1 < maxY; y1++) {
			for (int x1 = minX; x1 < minX + (int)(150 * scaleFactor); x1++) {
				cv::Scalar pixel = imgDiffSmall.at<uchar>(y1, x1);
				if (pixel.val[0] > 254) {

					if (armMinY > y1) {
						armMinY = y1;
					}
					else {
						armMinY = armMinY;
					}

					if (armMaxY < y1) {
						armMaxY = y1;
					}
					else {
						armMaxY = armMaxY;
					}

				}
			}
		}
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
	/*
	cv::Mat imgCanny;
	cv::Canny(img, imgCanny, 0, 255, 3);
	imshow("imgCanny", imgCanny);	
	
	cv::findContours(imgCanny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	vector<cv::Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		mu[i] = moments(contours[i], false);
	}
	vector<cv::Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	cv::RNG rng(12345);
	cv::Mat imgContours = cv::Mat::zeros(imgCanny.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(imgContours, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
		circle(imgContours, mc[i], 4, color, -1, 8, 0);
	}
	imshow("imgContours", imgContours);

	/*
	cv::Moments moments = cv::moments(img);
	double area = cv::contourArea(moments.m00);
	if (area > 10000) {
		persX0 = (int)(moments.m10 / area);
		persY0 = (int)(moments.m01 / area);
	} else {
		persX0 = -1;
		persY0 = -1;
	}
	*/
}

void searchGesture(cv::Mat frame) {
	cv::Mat img;
	frame.copyTo(img);

	/*
	//In HSV Farbraum umwandeln
	cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
	imshow("IMG-HSV", img);

	cv::Mat imgThreshold;
	inRange(img, cv::Scalar(0, 45, 0), cv::Scalar(15, 200, 200), imgThreshold);
	imshow("IMG-Hautfarbe", imgThreshold);
	*/

	cv::Mat imgDiff;
	cv::absdiff(bgImg, img, imgDiff);
	cv::cvtColor(imgDiff, imgDiff, CV_BGR2GRAY);
	cv::blur(imgDiff, imgDiff, cv::Size(12, 12));
	cv::threshold(imgDiff, imgDiff, 20, 255, CV_THRESH_BINARY);/**/
	imshow("diffImg", imgDiff);

	if (!searchPositions(imgDiff)) {
		return;
	}
	
	int currentPositions[POSITION_VARS] = { minX, minY, maxX, maxY, armMinY, armMaxX, armMaxY };
	if (positionsCount >= POSITIONS_ACCURACY) {
		positionsCount = 0;
	}
	for (int i = 0; i < POSITION_VARS; i++) {
		positions[positionsCount][i] = currentPositions[i];
	}

	int positionsAverageDelta[POSITION_VARS] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	int minus5 = positionsCount - AVERAGE;
	if (minus5 > -1) {
		for (int i = 0; i < POSITION_VARS; i++) {
			for (int j = minus5; j < positionsCount; j++) {
				if (j + 1 >= positionsCount) break;
				positionsAverageDelta[i] += positions[j+1][i] - positions[j][i];
			}
			positionsAverageDelta[i] = (int)(positionsAverageDelta[i] / ((float)AVERAGE - 1));
		}
	}
	else {
		int end = AVERAGE + minus5;
		int start = POSITIONS_ACCURACY + minus5;
		for (int i = 0; i < POSITION_VARS; i++) {
			for (int k = start; k < POSITIONS_ACCURACY; k++) {
				if (k + 1 >= POSITIONS_ACCURACY) {
					if (end > 0) {
						positionsAverageDelta[i] += positions[0][i] - positions[k][i];
					}
					break;
				}				
				positionsAverageDelta[i] += positions[k + 1][i] - positions[k][i];
			}
			for (int j = 0; j < end + 1; j++) {
				if (j + 1 >= end) break;
				positionsAverageDelta[i] += positions[j+1][i] - positions[j][i];
			}
			
			positionsAverageDelta[i] = (int)(positionsAverageDelta[i] / ((float)AVERAGE - 1));
		}
	}
	positionsCount++;

	if (positionsAverageDelta[0] > -80 && positionsAverageDelta[0] < -40) {
		gesture = KAMEHAMEHA;
	}	
	
	if (aniFrameNumber < 2) {
		kamehamehaX = armMinX;
		kamehamehaY = armMinY + (int)((armMaxY - armMinY) / 2);
	}
	else {
		int posCount = positionsCount - 1;
		if (posCount < 0) {
			posCount = POSITIONS_ACCURACY - 1;
			if (abs(positions[0][4] - positions[posCount][4]) < 30) {
				kamehamehaX = armMinX;
			}
		}
		else {
			if (abs(positions[posCount][4] - positions[positionsCount][4]) < 30) {
				kamehamehaX = armMinX;
			}
		}
	}

	
	/**/

}

/**********************************************************/



int MonoLoopOldStyle()
{
	IplImage* grabImage = 0;
	IplImage* resultImage = 0;
	int key;

	// create window for live video
	cvNamedWindow("Live", CV_WINDOW_AUTOSIZE);
	// create connection to camera
	CvCapture* capture = cvCaptureFromCAM(0);
	// init camera
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, MY_IMAGE_WIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, MY_IMAGE_HEIGHT);

	// check connection to camera by grabbing a frame
	if (!cvGrabFrame(capture))
	{
		cvReleaseCapture(&capture);
		cvDestroyWindow("Live");
		printf("Could not grab a frame\n\7");
		return -1;
	}

	// retrieve the captured frame
	grabImage = cvRetrieveFrame(capture);
	// init result image, e.g. with size and depth of grabImage
	resultImage = cvCreateImage(cvGetSize(grabImage), grabImage->depth, grabImage->nChannels);

	bool continueGrabbing = true;
	while (continueGrabbing)
	{
		if (!cvGrabFrame(capture))
		{
			cvReleaseCapture(&capture);
			cvDestroyWindow("Live");
			cvReleaseImage(&grabImage);
			printf("Could not grab a frame\n\7");
			return -1;
		}
		else
		{
			grabImage = cvRetrieveFrame(capture);

			/*******************************todo*****************************/
			cvCopy(grabImage, resultImage, NULL);
			/***************************end todo*****************************/

			cvShowImage("Live", resultImage);

			key = cvWaitKey(MY_WAIT_IN_MS);

			if (key == 27)
				continueGrabbing = false;
		}
	}

	// release all
	cvReleaseCapture(&capture);
	cvDestroyWindow("Live");
	cvReleaseImage(&resultImage);

	return 0;
}

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

		if (gesture == KAMEHAMEHA) {
			cv::Mat addImg = getAniImg(aniFrameNumber);
			if (kamehamehaX > -1 && kamehamehaX < addImg.cols && kamehamehaY > -1 && kamehamehaY < addImg.rows) {
				addImg = resizeAndPosAnimation(addImg, 1, kamehamehaX, kamehamehaY);
				cv::add(inputFrame, addImg, outputFrame);
			}
		}
		
		
		//if (kamehamehaGestureDiscovered(inputFrame)) {
			//cv::Mat addImg = getAniImg(aniFrameNumber);
			//if (!aniFinished && kamehamehaX > -1 && kamehamehaX < addImg.cols && kamehamehaY > -1 && kamehamehaY < addImg.rows) {				
				//addImg = resizeAndPosAnimation(addImg, 1, kamehamehaX, kamehamehaY);
				//cv::add(inputFrame, addImg, outputFrame);
			//}
			//L�st Endlosschleife der Animation aus
			/*
			else {
				aniFinished = false;
			}
			*/
		//}

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

int StereoLoop()
{
	cv::VideoCapture cap1(0);
	cv::VideoCapture cap2(1);

	if (!cap1.isOpened())
	{
		cout << "Cannot open the video cam [0]" << endl;
		return -1;
	}

	if (!cap2.isOpened())
	{
		cout << "Cannot open the video cam [1]" << endl;
		return -1;
	}

	// Set cameras to 15fps (if wanted!!!)
	cap1.set(CV_CAP_PROP_FPS, 15);
	cap2.set(CV_CAP_PROP_FPS, 15);

	double dWidth1 = cap1.get(CV_CAP_PROP_FRAME_WIDTH);
	double dHeight1 = cap1.get(CV_CAP_PROP_FRAME_HEIGHT);
	double dWidth2 = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
	double dHeight2 = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);

	// Set image size
	cap1.set(CV_CAP_PROP_FRAME_WIDTH, MY_IMAGE_WIDTH);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, MY_IMAGE_HEIGHT);
	cap2.set(CV_CAP_PROP_FRAME_WIDTH, MY_IMAGE_WIDTH);
	cap2.set(CV_CAP_PROP_FRAME_HEIGHT, MY_IMAGE_HEIGHT);

	// display the frame size that OpenCV has picked in order to check 
	cout << "cam[0] Frame size: " << dWidth1 << " x " << dHeight1 << endl;
	cout << "cam[1] Frame size: " << dWidth2 << " x " << dHeight2 << endl;
	cv::namedWindow("cam[0]", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("cam[1]", CV_WINDOW_AUTOSIZE);

	cv::Mat inputFrame1, inputFrame2;
	cv::Mat outputFrame1, outputFrame2;

	while (1)
	{

		bool bSuccess1 = cap1.read(inputFrame1);
		bool bSuccess2 = cap2.read(inputFrame2);

		if (!bSuccess1)
		{
			cout << "Cannot read a frame from video stream [0]" << endl;
			break;
		}

		if (!bSuccess2)
		{
			cout << "Cannot read a frame from video stream [1]" << endl;
			break;
		}


		/*******************************todo*****************************/
		outputFrame1 = inputFrame1;
		outputFrame2 = inputFrame2;
		/***************************end todo*****************************/


		imshow("cam[0]", outputFrame1);
		imshow("cam[1]", outputFrame2);

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

	clearMovements();

	return MonoLoop();
}
