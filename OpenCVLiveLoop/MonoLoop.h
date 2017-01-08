#pragma once
#include "highgui.h"
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;

class CMonoLoop
{
  int m_width;
  int m_height;

  cv::VideoCapture m_cap;
  cv::Mat m_inputFrame;
  cv::Mat m_outputFrame;
  
  bool m_isInit;

  // example function for manipulating pixels
  void MyInvert();

  int ProcessImage();

  int InitCam();
  int CreateImages();

public:
  CMonoLoop(void);
  ~CMonoLoop(void);

  int Run();
};

