#include "StdAfx.h"
#include "StereoLoop.h"


CStereoLoop::CStereoLoop(void)
{
  m_width  = 640;
  m_height = 480;
  m_isInit = false;
}


CStereoLoop::~CStereoLoop(void)
{

}


int CStereoLoop::CreateImages()
{
  m_outputFrame1 = m_inputFrame1;
  m_outputFrame2 = m_inputFrame2;

  // create additional images for processing here
  // return -1 in case of failure

  return 0;
}

int CStereoLoop::InitCam()
{
  m_cap1.open(0);
  if(!m_cap1.isOpened())
  {
    cout << "Cannot open the video cam [0]" << endl;
    return -1;
  }

  m_cap2.open(1);
  if(!m_cap2.isOpened())
  {
    cout << "Cannot open the video cam [1]" << endl;
    return -1;
  }
  
  // Set cameras to 15fps (if wanted!!!)
  m_cap1.set(CV_CAP_PROP_FPS, 15);
  m_cap2.set(CV_CAP_PROP_FPS, 15);

  double dWidth1 = m_cap1.get(CV_CAP_PROP_FRAME_WIDTH);
  double dHeight1 = m_cap1.get(CV_CAP_PROP_FRAME_HEIGHT);
  double dWidth2 = m_cap2.get(CV_CAP_PROP_FRAME_WIDTH);
  double dHeight2 = m_cap2.get(CV_CAP_PROP_FRAME_HEIGHT);

  // Set image size
  m_cap1.set(CV_CAP_PROP_FRAME_WIDTH, m_width);
  m_cap1.set(CV_CAP_PROP_FRAME_HEIGHT, m_height);
  m_cap2.set(CV_CAP_PROP_FRAME_WIDTH, m_width);
  m_cap2.set(CV_CAP_PROP_FRAME_HEIGHT, m_height);

  // display the frame size that OpenCV has picked in order to check 
  cout << "cam[0] Frame size: " << dWidth1 << " x " << dHeight1 << endl;
  cout << "cam[1] Frame size: " << dWidth2 << " x " << dHeight2 << endl;
  cv::namedWindow("cam[0]",CV_WINDOW_AUTOSIZE);
  cv::namedWindow("cam[1]",CV_WINDOW_AUTOSIZE);  

  bool bSuccess1 = m_cap1.read(m_inputFrame1);
  bool bSuccess2 = m_cap2.read(m_inputFrame2);

  if (!bSuccess1)
  {
    cout << "Cannot read a frame from video stream [0]" << endl;
    return -1;
  }

  if (!bSuccess2)
  {
    cout << "Cannot read a frame from video stream [1]" << endl;
    return -1;
  }

  if (CreateImages() != 0)
    return -1;

  m_isInit = true;

  return 0;
}

int CStereoLoop::Run()
{
  if (!m_isInit)
    InitCam();

  while(1)
  {
   
    bool bSuccess1 = m_cap1.read(m_inputFrame1);
    bool bSuccess2 = m_cap2.read(m_inputFrame2);

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
    
    if (ProcessImage() != 0)
    {
      cout << "Cannot process image" << endl;
      return -1;
    }

    imshow("cam[0]", m_outputFrame1);
    imshow("cam[1]", m_outputFrame2);

    if(cv::waitKey(30) == 27)
    {
      cout << "ESC key is pressed by user" << endl;
      break;
    }
  }
  return 0;
}

int CStereoLoop::ProcessImage()
{
  /// todo, e.g.
  m_outputFrame1 = m_inputFrame1;
  m_outputFrame2 = m_inputFrame2;
  // or


  return 0;
}


