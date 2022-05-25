#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "LaneDetection.h"

double img_size, img_center;
double left_m, right_m;
Point left_b, right_b;
bool left_detect = false, right_detect = false;

vector<vector<Vec4i>> separateLine(Mat img_edges, vector<Vec4i> lines);
vector<Point> regression(vector<vector<Vec4i>> separatedLines, Mat img_input);
Mat drawLine(Mat img_input, vector<Point> lane);

Mat lane_detection(Mat img) {

	Mat HSV;

	cvtColor(img, HSV, COLOR_BGR2HSV); //Color space를 BGR에서 HSV로 변경 (HSV에 저장)
	// Scalar Class: 영상의 픽셀값을 표현하는 용도로 사용
	// Hue(0~179), Saturation(0~255), Value(200~255)
	Scalar lower_white = Scalar(0, 0, 200);
	Scalar upper_white = Scalar(180, 255, 255);
	inRange(HSV, lower_white, upper_white, HSV); // inRange함수: 범위 안에 드는 색만 검출 (HSV에 저장)
	// 흰색에 가까운 픽셀값 찾아냄

	Mat bilateral_Filter, dilate_filter, canny;
	bilateralFilter(HSV, bilateral_Filter, 5, 100, 100); // 검출된 흰색 픽셀을 더 뚜렷하게 나타내기 + 노이즈제거
	dilate(bilateral_Filter, dilate_filter, Mat()); //흰색 픽셀을 팽창
	Canny(dilate_filter, canny, 150, 255); // 흰색 픽셀의 가장자리 검출
	//imshow("img_combine img", HSV);
	//imshow("canny img", canny);

	//ROI
	Point point[4];
	point[0] = Point(400, 300);
	point[1] = Point(220, 420);
	point[2] = Point(650, 420);
	point[3] = Point(440, 300);

	Mat img_mask = Mat::zeros(canny.rows, canny.cols, CV_8UC1);
	Mat ROI = Mat::zeros(canny.rows, canny.cols, CV_8UC1);

	Scalar ignore_mask_color = Scalar(255, 255, 255);
	const Point* ppt[1] = { point };
	int npt[] = { 4 };

	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);

	bitwise_and(canny, img_mask, ROI);
	//imshow("img_mask", img_mask);
	//imshow("ROI", ROI);

	Mat lineResult;
	cvtColor(ROI, lineResult, COLOR_GRAY2BGR); //이미지의 타입을 통일해야하기 때문에 BGR 포맷으로 바꿈 (lineResult에 저장)

	vector<Vec4i> lines; //Vec4i 클래스는 int 자료형 네 개를 저장할 수 있는 OpenCV 벡터 클래스
	HoughLinesP(ROI, lines, 1, CV_PI / 180, 30, 10, 20);

	//HoughLine Transform을 통해 검출된 edge들 중
	//진짜 차선을 검출하고 이를 line에 저장한다.

	//Mat img_lines;
    Mat img_result;

	vector<vector<Vec4i> > separated_lines;
	vector<Point> lane;

	if (lines.size() > 0) {
		//추출한 직선성분으로 좌우 차선에 있을 가능성이 있는 직선들만 따로 뽑아서 좌우 각각 직선을 계산한다. 
		separated_lines = separateLine(ROI, lines);

		//선형 회귀를 하여 가장 적합한 선을 찾는다.
		lane = regression(separated_lines, img);

		// 좌우 차선을 영상에 선으로 그린다.
		img_result = drawLine(img, lane);
	}
	return img_result;
}


vector<vector<Vec4i>> separateLine(Mat img_edges, vector<Vec4i> lines) {

	//검출된 모든 허프변환 직선들을 기울기 별로 정렬한다.
	//선을 기울기와 대략적인 위치에 따라 좌우로 분류한다.

	vector<vector<Vec4i>> output(2);
	Point ini, fini;
	vector<double> slopes;
	vector<Vec4i> selected_lines, left_lines, right_lines; // Vec4i 클래스는 int 자료형 네 개를 저장할 수 있는 OpenCV 벡터 클래스
	double slope_thresh = 0.3;

	//검출된 직선들의 기울기를 계산
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		ini = Point(line[0], line[1]);
		fini = Point(line[2], line[3]);

		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y))
			/ (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001); //기울기 계산 y증가량/x증가량

		//기울기가 너무 수평인 선은 제외
		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope); //push_back : slope를 slopes vector의 끝에 요소로 추가
			selected_lines.push_back(line);  //line을 selected_lines vector의 끝에 요소로 추가
		}
	}

	//선들을 좌우 선으로 분류
	img_center = static_cast<double>((img_edges.cols / 2));
	for (int i = 0; i < selected_lines.size(); i++) {
		ini = Point(selected_lines[i][0], selected_lines[i][1]);
		fini = Point(selected_lines[i][2], selected_lines[i][3]);

		if (slopes[i] > 0 && fini.x > img_center && ini.x > img_center) {
			right_lines.push_back(selected_lines[i]);
			right_detect = true;
		}
		else if (slopes[i] < 0 && fini.x < img_center && ini.x < img_center) {
			left_lines.push_back(selected_lines[i]);
			left_detect = true;
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}

vector<Point> regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
	// separatedLines : 분류된 좌우 선
	// 선형 회귀를 통해 좌우 차선 각각의 가장 적합한 선을 찾는다. 
	// 선형 회귀 : 주어진 데이터로부터 y 와 x 의 관계를 가장 잘 나타내는 직선을 그리는 일

	vector<Point> output(4);
	Point ini, fini;
	Point ini2, fini2;
	Vec4d left_line, right_line; //4개의 double요소
	vector<Point> left_pts, right_pts;

	if (right_detect) {
		for (auto i : separatedLines[0]) { //output[0] = right_lines
			ini = Point(i[0], i[1]);
			fini = Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}

		if (right_pts.size() > 0) {
			//주어진 contour에 최적화된 직선 추출
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01); 
			//cv:fitLine 함수는 입력된 포인트에서 직선성분을 찾아주는 함수
			// 단위 방향 벡터(right_line 첫 두개 값), 선에 놓인 한 점의 좌표(right_line 마지막 두 값) 형태인 선 방정식의 파라미터를 제공
			// 마지막 두 파라미터는 선 파라미터에 대한 요구 정확도를 지정

			right_m = right_line[1] / right_line[0];  //기울기
			right_b = Point(right_line[2], right_line[3]); //선위의 한 점
		}
	}

	if (left_detect) {
		for (auto j : separatedLines[1]) { //output[1] = left_lines
			ini2 = Point(j[0], j[1]);
			fini2 = Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}

		if (left_pts.size() > 0) {
			//주어진 contour에 최적화된 직선 추출
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //기울기
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	//좌우 선 각각의 두 점을 계산한다.
	//y = m*x + b  --> x = (y-b) / m
	int ini_y = img_input.rows;
	int fin_y = 330;

	double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
	double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

	double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
	double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_ini_x, ini_y);
	output[1] = Point(right_fin_x, fin_y);
	output[2] = Point(left_ini_x, ini_y);
	output[3] = Point(left_fin_x, fin_y);

	// 좌우선의 좌표 2개씩 저장

	return output;
}

Mat drawLine(Mat img_input, vector<Point> lane) {
	//lane : 좌우선의 좌표 2개씩 저장 (총 4개 좌표)
	// 좌우 차선을 영상에 선으로 그린다.

	line(img_input, lane[0], lane[1], Scalar(0, 0, 255), 5, LINE_AA);
	line(img_input, lane[2], lane[3], Scalar(0, 0, 255), 5, LINE_AA);

	return img_input;
}


