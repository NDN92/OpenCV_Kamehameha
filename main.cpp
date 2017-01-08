#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {

  // The indices are 1 and 2 since 0 is my built-in webcam (I'm using a notebook)
  VideoCapture cap1(1);
  VideoCapture cap2(2);

  if(!cap1.isOpened())
  {
    cout << "Cannot open the video cam [1]" << endl;
    return -1;
  }

  if(!cap2.isOpened())
  {
    cout << "Cannot open the video cam [2]" << endl;
    return -1;
  }

  // Set both cameras to 15fps
  cap1.set(CV_CAP_PROP_FPS, 15);
  cap2.set(CV_CAP_PROP_FPS, 15);

  double dWidth1 = cap1.get(CV_CAP_PROP_FRAME_WIDTH);
  double dHeight1 = cap1.get(CV_CAP_PROP_FRAME_HEIGHT);
  double dWidth2 = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
  double dHeight2 = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);

  // Values taken from output of Version 1 and used to setup the exact same parameters with the exact same values!
  // ERROR: using set() with the values retrieved with get() cause error (on Linux)
//  cap1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
//  cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//  cap2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
//  cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  // Here I display the frame size that OpenCV has picked for me - it is 640x480 for both cameras
  cout << "cam[1] Frame size: " << dWidth1 << " x " << dHeight1 << endl;
  cout << "cam[2] Frame size: " << dWidth2 << " x " << dHeight2 << endl;
  namedWindow("cam[1]",CV_WINDOW_AUTOSIZE);
  namedWindow("cam[2]",CV_WINDOW_AUTOSIZE);

  while(1)
  {
    Mat frame1, frame2;
    bool bSuccess1 = cap1.read(frame1);
    bool bSuccess2 = cap2.read(frame2);

    if (!bSuccess1)
    {
      cout << "Cannot read a frame from video stream [1]" << endl;
      break;
    }

    if (!bSuccess2)
    {
      cout << "Cannot read a frame from video stream [2]" << endl;
      break;
    }

    imshow("cam[1]", frame1);
    imshow("cam[2]", frame2);

    if(waitKey(30) == 27)
    {
      cout << "ESC key is pressed by user" << endl;
      break;
    }
  }

  return 1;
}
