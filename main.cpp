#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "LaneDetection.h"


int main() {
	VideoCapture cap("video_Test2.mp4");

	if (!cap.isOpened()) {
		printf("Can't open the camera");
		return -1;
	}

	while (1) {
		Mat img;
		Mat img_result;
		cap >> img;

		if (img.empty()) {
			printf("empty image");
			return -1;
		}

		img_result = lane_detection(img);

		imshow("result", img_result);

		if (waitKey(1) == 27)
			break;
	}

	return 0;
	
}

