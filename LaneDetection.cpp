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

	cvtColor(img, HSV, COLOR_BGR2HSV); //Color space�� BGR���� HSV�� ���� (HSV�� ����)
	// Scalar Class: ������ �ȼ����� ǥ���ϴ� �뵵�� ���
	// Hue(0~179), Saturation(0~255), Value(200~255)
	Scalar lower_white = Scalar(0, 0, 200);
	Scalar upper_white = Scalar(180, 255, 255);
	inRange(HSV, lower_white, upper_white, HSV); // inRange�Լ�: ���� �ȿ� ��� ���� ���� (HSV�� ����)
	// ����� ����� �ȼ��� ã�Ƴ�

	Mat bilateral_Filter, dilate_filter, canny;
	bilateralFilter(HSV, bilateral_Filter, 5, 100, 100); // ����� ��� �ȼ��� �� �ѷ��ϰ� ��Ÿ���� + ����������
	dilate(bilateral_Filter, dilate_filter, Mat()); //��� �ȼ��� ��â
	Canny(dilate_filter, canny, 150, 255); // ��� �ȼ��� �����ڸ� ����
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
	cvtColor(ROI, lineResult, COLOR_GRAY2BGR); //�̹����� Ÿ���� �����ؾ��ϱ� ������ BGR �������� �ٲ� (lineResult�� ����)

	vector<Vec4i> lines; //Vec4i Ŭ������ int �ڷ��� �� ���� ������ �� �ִ� OpenCV ���� Ŭ����
	HoughLinesP(ROI, lines, 1, CV_PI / 180, 30, 10, 20);

	//HoughLine Transform�� ���� ����� edge�� ��
	//��¥ ������ �����ϰ� �̸� line�� �����Ѵ�.

	//Mat img_lines;
    Mat img_result;

	vector<vector<Vec4i> > separated_lines;
	vector<Point> lane;

	if (lines.size() > 0) {
		//������ ������������ �¿� ������ ���� ���ɼ��� �ִ� �����鸸 ���� �̾Ƽ� �¿� ���� ������ ����Ѵ�. 
		separated_lines = separateLine(ROI, lines);

		//���� ȸ�͸� �Ͽ� ���� ������ ���� ã�´�.
		lane = regression(separated_lines, img);

		// �¿� ������ ���� ������ �׸���.
		img_result = drawLine(img, lane);
	}
	return img_result;
}


vector<vector<Vec4i>> separateLine(Mat img_edges, vector<Vec4i> lines) {

	//����� ��� ������ȯ �������� ���� ���� �����Ѵ�.
	//���� ����� �뷫���� ��ġ�� ���� �¿�� �з��Ѵ�.

	vector<vector<Vec4i>> output(2);
	Point ini, fini;
	vector<double> slopes;
	vector<Vec4i> selected_lines, left_lines, right_lines; // Vec4i Ŭ������ int �ڷ��� �� ���� ������ �� �ִ� OpenCV ���� Ŭ����
	double slope_thresh = 0.3;

	//����� �������� ���⸦ ���
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		ini = Point(line[0], line[1]);
		fini = Point(line[2], line[3]);

		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y))
			/ (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001); //���� ��� y������/x������

		//���Ⱑ �ʹ� ������ ���� ����
		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope); //push_back : slope�� slopes vector�� ���� ��ҷ� �߰�
			selected_lines.push_back(line);  //line�� selected_lines vector�� ���� ��ҷ� �߰�
		}
	}

	//������ �¿� ������ �з�
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
	// separatedLines : �з��� �¿� ��
	// ���� ȸ�͸� ���� �¿� ���� ������ ���� ������ ���� ã�´�. 
	// ���� ȸ�� : �־��� �����ͷκ��� y �� x �� ���踦 ���� �� ��Ÿ���� ������ �׸��� ��

	vector<Point> output(4);
	Point ini, fini;
	Point ini2, fini2;
	Vec4d left_line, right_line; //4���� double���
	vector<Point> left_pts, right_pts;

	if (right_detect) {
		for (auto i : separatedLines[0]) { //output[0] = right_lines
			ini = Point(i[0], i[1]);
			fini = Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}

		if (right_pts.size() > 0) {
			//�־��� contour�� ����ȭ�� ���� ����
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01); 
			//cv:fitLine �Լ��� �Էµ� ����Ʈ���� ���������� ã���ִ� �Լ�
			// ���� ���� ����(right_line ù �ΰ� ��), ���� ���� �� ���� ��ǥ(right_line ������ �� ��) ������ �� �������� �Ķ���͸� ����
			// ������ �� �Ķ���ʹ� �� �Ķ���Ϳ� ���� �䱸 ��Ȯ���� ����

			right_m = right_line[1] / right_line[0];  //����
			right_b = Point(right_line[2], right_line[3]); //������ �� ��
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
			//�־��� contour�� ����ȭ�� ���� ����
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //����
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	//�¿� �� ������ �� ���� ����Ѵ�.
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

	// �¿켱�� ��ǥ 2���� ����

	return output;
}

Mat drawLine(Mat img_input, vector<Point> lane) {
	//lane : �¿켱�� ��ǥ 2���� ���� (�� 4�� ��ǥ)
	// �¿� ������ ���� ������ �׸���.

	line(img_input, lane[0], lane[1], Scalar(0, 0, 255), 5, LINE_AA);
	line(img_input, lane[2], lane[3], Scalar(0, 0, 255), 5, LINE_AA);

	return img_input;
}


