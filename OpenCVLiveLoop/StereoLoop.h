#pragma once
#include "highgui.h"
#include "cv.h"
#include <iostream>
using namespace std;

class CStereoLoop
{
  int m_width;
  int m_height;

  cv::VideoCapture m_cap1, m_cap2;
  cv::Mat m_inputFrame1, m_inputFrame2;
  cv::Mat m_outputFrame1, m_outputFrame2;
  
  bool m_isInit;

  int ProcessImage();

  int InitCam();
  int CreateImages();
public:
  CStereoLoop(void);
  ~CStereoLoop(void);

  int Run();
};
  
