#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
//#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;


int binarization(Mat& src, Mat& dst, int threshold);
int grassfire(Mat& src, Mat& dst);
int segmentation(Mat& src, Mat& dst);
int viewSegments(Mat& src, Mat& dst, int regID);
int moments(Mat& src, Mat& dst);
int Sort_Descending(int* A, int* B, int length);
int append_image_data_csv(char* filename, char* image_filename, std::vector<float>& image_data, int reset_file);
int blur5x5(cv::Mat& src, cv::Mat& dst);
int calc_grey_histogram(cv::Mat& src, cv::Mat& hist);
char* extract_features(cv::Mat& src,  cv::Mat& src2, int save_in_csv);
int feature_matching(cv::Mat& src, int reg_id);
int read_image_data_csv(char* filename, std::vector<char*>& filenames, std::vector<std::vector<float>>& data, int echo_file);
int append_image_data_csv(char* filename, char* image_filename, std::vector<float>& image_data, int reset_file);

float distance(std::vector<float>  x, std::vector<float>  y);
float calculateSD(std::vector<float> data);

