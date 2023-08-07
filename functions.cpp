/*
* CS5330: Assignment 3
Author: RAVI MAHESHWARI
NUID:002104786
*/


#include "functions.h"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
//#include "read_from_csv.cpp"
//#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
#define FOREGROUND 0
#define BACKGROUND 255
int AREAS[100] = { 0 };
int REGIONS[100] = { 0 };
/*
Creates a black and white image out of 3 channel RGB input
Output is a single channel image
*/
int binarization(Mat& src, Mat& dst, int threshold)
{
	//src.copyTo(dst);
	int dim[2] = { src.rows, src.cols };
	dst = cv::Mat::zeros(2,dim, CV_8U);
	int thr = 0;
	cv::Mat hist;

	thr = calc_grey_histogram(src, hist);
	for (int i = 0;i < src.rows;i++)
		for (int j = 0;j < src.cols;j++)
		{
			//for (int c = 0;c < 3;c++)
			//int gamma = 50;
			//if ((i < gamma)  || (j < gamma) || (j > src.cols - gamma) || (i > src.rows - gamma))
			//{
			//	thr = threshold + 50;
			//}

			
			if (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2] >= thr)
			{
				dst.at<uchar>(i, j) = BACKGROUND;
				//dst.at<Vec3b>(i, j)[1] = BACKGROUND;
				//dst.at<Vec3b>(i, j)[2] = BACKGROUND;
			}
			else
			{
				dst.at<uchar>(i, j) = FOREGROUND;
				//dst.at<Vec3b>(i, j)[1] = FOREGROUND;
				//dst.at<Vec3b>(i, j)[2] = FOREGROUND;
			}
		}
	convertScaleAbs(dst, dst);

	return(0);
}

/*
Finds distances of individual pixels from the background 
Input and Output are 1 channel a one channel image 
*/
int grassfire(Mat& src, Mat& dst)
{
	
	dst = cv::Mat::zeros(src.size(), CV_8U);

	//Pass 1 : Left to Right, Up to Down
	for (int i = 1;i < src.rows;i++)
	{
		for (int j = 1;j < src.cols;j++)
		{
			if (src.at<uchar>(i, j) == FOREGROUND)
			{
				if ((src.at<uchar>(i - 1, j) == BACKGROUND) || (src.at<uchar>(i, j - 1) == BACKGROUND))
				{
					dst.at<uchar>(i, j) = 1;
				}
				else
				{
					dst.at<uchar>(i, j)= min(dst.at<uchar>(i-1, j), dst.at<uchar>(i, j-1))+1;
				}
			}
		}
	}


	//Pass 2: RIGHT to LEFT, BOTTOM to TOP
	for (int i = src.rows -2;i >0 ;i--)
	{
		for (int j = src.cols-2;j >0 ;j--)
		{
			if (src.at<uchar>(i, j)== FOREGROUND)
			{
				if ((src.at<uchar>(i + 1, j) == BACKGROUND) || (src.at<uchar>(i, j + 1) == BACKGROUND))
				{
					dst.at<uchar>(i, j) = 1;
				}
				else
				{
					uchar a =  min(dst.at<uchar>(i + 1, j), dst.at<uchar>(i, j + 1)) + 1;
					dst.at<uchar>(i, j) = min(a, dst.at<uchar>(i, j));
				}
			}
		}
	}

	//for accurate distance transform representation - Not really necessary
	for (int i = 0;i < src.rows;i++)
		for (int j = 0;j < src.cols;j++)
	//		for (int c = 0;c < 3;c++)
				dst.at<uchar>(i, j) = dst.at<uchar>(i, j)*10;

	convertScaleAbs(dst, dst);
	
	return(0);
}


/*
Function to assign region IDs to the segment an image and assign region IDs to each segment
Input is the binary image
*/
int segmentation(Mat& src, Mat& dst)
{
	uchar regionID = 0;
	dst = cv::Mat::zeros(src.size(), CV_8U);
	//Pass 1 : Left to Right, Up to Down
	for (int i = 1;i < src.rows;i++)
	{
		for (int j = 1;j < src.cols;j++)
		{
			if (src.at<uchar>(i, j) != BACKGROUND)
			{
				if ((src.at<uchar>(i - 1, j) != BACKGROUND) || (src.at<uchar>(i, j - 1) != BACKGROUND))
				{
					if (dst.at<uchar>(i - 1, j) > 0 && dst.at<uchar>(i, j - 1) > 0)
					{
						//Union of the regions
						dst.at<uchar>(i, j) = min(dst.at<uchar>(i - 1, j), dst.at<uchar>(i, j - 1));
						dst.at<uchar>(i - 1, j) = dst.at<uchar>(i, j);
						dst.at<uchar>(i, j - 1) = dst.at<uchar>(i, j);
					}
					else if (dst.at<uchar>(i - 1, j) > 0 || dst.at<uchar>(i, j - 1) > 0)
					{
						dst.at<uchar>(i, j) = max(dst.at<uchar>(i - 1, j), dst.at<uchar>(i, j - 1));
					}
					else
					{
						dst.at<uchar>(i, j) = max(dst.at<uchar>(i, j), ++regionID);
					}

				}
				else
				{
					dst.at<uchar>(i, j) = max(dst.at<uchar>(i, j), ++regionID);
				}
			}
		}
	}

	//Pass 2: RIGHT to LEFT, BOTTOM to TOP
	for (int i = src.rows - 2;i > 0;i--)
	{
		for (int j = src.cols - 2;j > 0;j--)
		{
			if (src.at<uchar>(i, j) != BACKGROUND)
			{
				if ((src.at<uchar>(i + 1, j) != BACKGROUND) || (src.at<uchar>(i, j + 1) != BACKGROUND))
				{
					if (dst.at<uchar>(i + 1, j) > 0 && dst.at<uchar>(i, j + 1) > 0)
					{
						dst.at<uchar>(i, j) = min(dst.at<uchar>(i + 1, j), dst.at<uchar>(i, j + 1));
						dst.at<uchar>(i + 1, j) = dst.at<uchar>(i, j);
						dst.at<uchar>(i, j + 1) = dst.at<uchar>(i, j);
					}
				}
			}
		}
	}
	convertScaleAbs(dst, dst);
	return(0);
}

/*
Input : binay image/ image with regiod IDs
Output: colorful image showing each region in a different color 
*/
int viewSegments(Mat& src, Mat& dst, int regID)
{
	//int dim[3] = {src.rows, src.cols, 3};
	Mat dst2(src.size(), CV_8UC3);
	//dst = cv::Mat::zeros(3,dim, CV_16SC3);
	//dst = cv::Mat::zeros(3, dim, CV_8UC3);
	//src.copyTo(dst);
	for (int i = 1;i < src.rows;i++)
	{
		for (int j = 1;j < src.cols;j++)
		{
			if (regID == -1)
			{
				if (src.at<uchar>(i, j) <= 100)
				{
					for (int c = 0;c < 3;c++)
						dst2.at<Vec3b>(i, j)[c] = 100 * c + 20 * src.at<uchar>(i, j);
				}
				else
				{
					for (int c = 0;c < 3;c++)
						dst2.at<Vec3b>(i, j)[c] = 0;
				}
			}
			else
			{ 
				if (src.at<uchar>(i, j) == regID)
				{
					dst2.at<Vec3b>(i, j)[2] = 255;
				}
				else
				{
					for (int c = 0;c < 3;c++)
						dst2.at<Vec3b>(i, j)[c] = 0;
				}
			}


		}
	}
	//dst2.copyTo(dst);
	dst = dst2;
	convertScaleAbs(dst, dst);
	return(0);
}

/*
Input : image with region IDs
Otput : global variables M00, M01, M10, M11 are updated with moments
*/
int moments(Mat& src, Mat& dst)
{
	float val = 0;
	long x_centroid = 0, y_centroid = 0, sum = 0, x_max = 0, x_min = src.rows, y_min = src.cols, y_max = 0;
	long X_moment=0, Y_moment=0, M11=0;
	double mue01, mue10,mue11,alpha;
	int regionID;
	src.copyTo(dst);
	//std::vector<std::pair<int, int>> AreaVec;
	// 
	//Area calculation of top 10 regions
	

	for (int i = 0;i < 100;i++) { REGIONS[i] = i; AREAS[i] = 0; }

	/*Zeroth Moment*/
	for (int i = 0;i < src.rows;i++)
	{
		for (int j = 0;j < src.cols;j++)
		{
			regionID = src.at<uchar>(i, j);
			if ((regionID >= 0) && (regionID < 100)) //Region ID 0 corresponds to the background
			{
				AREAS[regionID]++;
			}

		}
	}
	/*Sort Areas based on size*/
	//printf("AREAS before");
	//for (int q = 0;q < 10;q++)
	//	printf("\nA %d", AREAS[q]);
	//printf("REGIONS before");
	//for (int q = 0;q < 10;q++)
	//	printf("\nA %d", REGIONS[q]);
	Sort_Descending(AREAS,REGIONS, 100);
	//printf("\n\nAREAS after");
	//for (int q = 0;q < 10;q++)
	//	printf("\nA %d", AREAS[q]);
	//printf("\n\nREGIONS after");
	//for (int q = 0;q < 10;q++)
	//	printf("\nA %d", REGIONS[q]);

	/*Plot top 1 regions*/
	for (int reg = 0; reg < 2;reg++) /**/
	{
		x_centroid = 0, y_centroid = 0, sum = 1, x_max = 0, x_min = src.rows, y_min = src.cols, y_max = 0;
		X_moment = 0, Y_moment = 0, M11 = 0;

		for (int i = 0;i < src.rows;i++)
		{
			for (int j = 0;j < src.cols;j++)
			{
				regionID = src.at<uchar>(i, j);
				if ((regionID > 0) && (regionID == REGIONS[reg])) //Region ID 0 corresponds to the background
				{

					x_centroid += i;
					y_centroid += j;
					sum++;
					if (i < x_min) x_min = i;
					if (j < y_min) y_min = j;
					if (i > x_max) x_max = i;
					if (j > y_max) y_max = j;
					dst.at<uchar>(i, j) = reg*20;
				}

			}
		}


		/*Centroids*/

		x_centroid = x_centroid / sum;
		y_centroid = y_centroid / sum;
		//printf("\n  %d Xmin %d Xmax%d Ymin%d Ymax%d", REGIONS[reg], x_min, x_max, y_min, y_max);
		//printf("\n X_center %d, Y_center %d", x_centroid, y_centroid);

		/*First Moment*/

		for (int i = 0;i < src.rows;i++)
		{
			for (int j = 0;j < src.cols;j++)
			{
				regionID = src.at<uchar>(i, j);
				if (regionID == REGIONS[reg])
				{
					X_moment += (i - x_centroid) * (i - x_centroid);
					Y_moment += (j - y_centroid) * (j - y_centroid);
					M11 += (i - x_centroid) * (j - y_centroid);
				}

			}
		}
		mue01 = X_moment / sum;
		mue10 = Y_moment / sum;
		mue11 = M11 / sum;

		alpha = 0.5 * (atan2((2 * mue11), (mue01 - mue10)));
		//printf("ALPHA = %f", alpha);
		/*Bounding box*/

		for (int i = x_min;i < x_max;i++)
		{
			dst.at<uchar>(i, y_min) = 255;
			dst.at<uchar>(i, y_max) = 255;
		}
		for (int j = y_min;j < y_max;j++)
		{
			dst.at<uchar>(x_min, j) = 255;
			dst.at<uchar>(x_max, j) = 255;
		}

		///*Axis of least moment*/
		for (double x = x_centroid, y = y_centroid, r = 0;y < y_max && x < x_max && r<100;x = x_centroid - r * cos(alpha), y = y_centroid - r * sin(alpha), r++)
		{
			if((x>0)&&(y>0)&&(x<dst.rows) && (y<dst.cols))
			dst.at<uchar>(x, y) = 125;
		}
		/*Region info: commented as 
		1.regiogs are now auto-identified (1st/2st largest): whichever isn't the back ground - wont work for multiple objects? 
		2.it was interfering with tag*/
		//Point text_position( y_centroid, x_centroid);
		////printf("TEXT for %d src cols %d src rows %d", REGIONS[reg], dst.cols, dst.rows);
		//int font_size = 1;
		//Scalar font_Color(123);//Declaring the color of the font//
		//int font_weight = 2;//Declaring the font weight//
		//auto name = std::to_string(REGIONS[reg]);
		////name[0] = strcat(char(reg),"hi");
		////strcpy(name, string(reg));
		////strcpy(name, "%s");
		//
		//putText(dst, name, text_position, FONT_HERSHEY_COMPLEX, font_size, font_Color, font_weight);//Putting the text in the matrix//

	}
	return(0);
		
}

/*
* Function Description
* Sorts a list <A> in descending order
* Also update a list <B> of IDs. to track which indexes are where after the update
*/
int Sort_Descending(int* A,int* B, int length)
{
	for(int i=0; i<length;i++ )
		for (int j=1; j<=i;j++)
			if (A[j] > A[j-1])
			{
				int tmp = A[j];
				A[j] = A[j - 1];
				A[j - 1] = tmp;
				tmp = B[j];
				B[j] = B[j - 1];
				B[j - 1] = tmp;
			}
	return(0);
}


/*
#############Function Description#################
Input : source image
Output: destination image
Return value: Returns zero
Funactionality:  if the user types 'b' it displays a gaussian blurred version of the image instead of color.
Here, the gaussian blur was performed by multiplying the image by [1 2 4 2 1] horizontally and then vertically.
###################################################
*/
int blur5x5(cv::Mat& src, cv::Mat& dst)
{
	int g[5] = { 3, 4, 5, 7, 3 };
	cv::Mat temp;
	src.copyTo(dst);
	//std::cout << "what is h " << sizeof(g)/sizeof(int) << std::endl;


	//apply_filter(temp, dst, g, 1);

	//apply_filter(temp, dst, g, 1);
	//apply_filter(temp, dst, g, 'c');
	//dst = temp;

	dst.convertTo(dst, CV_16SC3);
	src.convertTo(src, CV_16SC3);
	/*apply_filter(src, dst, g, 0);
	dst.copyTo(temp);
	temp.convertTo(temp, CV_16SC3);
	apply_filter(temp, dst, g, 1);*/

	for (int i = 2; i < src.rows - 2; i = i + 1)
	{

		for (int j = 0; j < src.cols; j = j + 1)
		{
			for (int c = 0; c < 3; c = c + 1)
			{
				dst.at<cv::Vec3s>(i, j)[c] = 1 * src.at<cv::Vec3s>(i - 2, j)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 2 * src.at<cv::Vec3s>(i - 1, j)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 4 * src.at<cv::Vec3s>(i, j)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 2 * src.at<cv::Vec3s>(i + 1, j)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 1 * src.at<cv::Vec3s>(i + 2, j)[c];
				dst.at<cv::Vec3s>(i, j)[c] = dst.at<cv::Vec3s>(i, j)[c] / 10;
			}


		}
	}

	for (int i = 0; i < src.rows; i = i + 1)
	{
		for (int j = 2; j < src.cols - 2; j = j + 1)
		{
			for (int c = 0; c < 3; c = c + 1)
			{
				dst.at<cv::Vec3s>(i, j)[c] = 5 * src.at<cv::Vec3s>(i, j - 2)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 2 * src.at<cv::Vec3s>(i, j - 1)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 4 * src.at<cv::Vec3s>(i, j)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 2 * src.at<cv::Vec3s>(i, j + 1)[c];
				dst.at<cv::Vec3s>(i, j)[c] += 1 * src.at<cv::Vec3s>(i, j + 2)[c];
				dst.at<cv::Vec3s>(i, j)[c] = dst.at<cv::Vec3s>(i, j)[c] / 10;
			}


		}
	}

	convertScaleAbs(dst, dst);

	return(0);
}

int calc_grey_histogram(cv::Mat& src, cv::Mat& hist)
{

	const int H_size = 10;
	const int divisor = 256 / H_size;
	int i, j, k;
	int dim[1] = { H_size};

	hist = cv::Mat::zeros(1, dim, CV_32S);
	//printf("Image rows:%d Image cols: %d", src.rows, src.cols);
	for (i = 0; i < src.rows; i++)
	{
		cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
		for (j = 0;j < src.cols;j++)
		{
			int r = sptr[j][2] / divisor;
			int g = sptr[j][1] / divisor;
			int b = sptr[j][0] / divisor;
			hist.at<int>((r+g+b)/3)++;
			//printf("RGB %d %d %d",r,g,b);
		}
	}

	//printf("Histogram\n");
	int grey_wind = 0;
	int thr = 3;
	int low = 99999;
	int peak2=0,peak1 = 0;
	for (i = 0;i < H_size;i++)
	{
		//printf("H%d  %d \n",i, hist.at<int>(i));
		grey_wind = grey_wind + hist.at<int>(i); /*is just num pixels at the end*/
		low = min(low, hist.at<int>(i));
		//if (grey_wind > (src.rows * src.cols) / 2) /*More than half the pixels are past*/
		//{
		//	if (hist.at<int>(i)>low) thr = i-1 ; /*Look for increase - then take peak*/
		//}
		if (hist.at<int>(i) > hist.at<int>(peak1))
		{
			peak2 = peak1; /*second highest peak*/
			peak1 = i; /*highest peak*/
		}
		else if(hist.at<int>(i) > hist.at<int>(peak2))
		{
			peak2 = i; /*second highest peak*/
		}
		else
		{/*do nothing*/ }

	}
	//printf("\Peak1: %d, Peak2: %d",peak1, peak2);
	if(peak1>peak2)
	{
		if ((peak1 - peak2) <= 2) peak2 += 2;
	}
	else
	{ 
		if ((peak2 - peak1) <= 2) peak1 += 2;
	}
	thr = ((peak1+peak2)*divisor*3) / 2;// thr * divisor; /*scale up the threashold*/
	thr = min(600,thr); // limiti threshold between plausible values
	thr = max(150, thr);
	//printf("Threshold: %d", thr);
	return thr;
}

char* extract_features(cv::Mat& src, cv::Mat& src2, int save_in_csv)
{
	long x_centroid = 0, y_centroid = 0, sum = 0, x_max = 0, x_min = src.rows, y_min = src.cols, y_max = 0;
	long X_moment = 0, Y_moment = 0, M11 = 0;
	double mue01, mue10, mue11, alpha;
	char filename[255], image_filename[255];
	std::vector<float>  x = { 0.0, 0.0, 0.0, 0.0, 0.0 };
	/*Storing features into CSV*/
	strcpy(filename, "C:/Northeastern/PRCV/visual C/ObjectRecognition/ObjectRecognition/Features.csv");
	strcpy(image_filename, "watedver");
	/*Now take region input from the user, and store associated features into the csv*/
	char Reg[2], Lab[25];
	int reg_id;
	//std::cout << "Enter the Region ID of interest!";
	//std::cin >> Reg;
	//reg_id = atoi(Reg);
	reg_id = REGIONS[0];
	if (reg_id == 0)reg_id = REGIONS[1];
	//printf("Chosen region: %d", reg_id);



		x_centroid = 0, y_centroid = 0, sum = 1, x_max = 0, x_min = src.rows, y_min = src.cols, y_max = 0;
		X_moment = 0, Y_moment = 0, M11 = 0;

		for (int i = 0;i < src.rows;i++)
		{
			for (int j = 0;j < src.cols;j++)
			{

				if (src.at<uchar>(i, j) == reg_id) //Region ID 0 corresponds to the background
				{

					x_centroid += i;
					y_centroid += j;
					sum++;
					if (i < x_min) x_min = i;
					if (j < y_min) y_min = j;
					if (i > x_max) x_max = i;
					if (j > y_max) y_max = j;
				}

			}
		}


		/*Centroids*/

		x_centroid = x_centroid / sum;
		y_centroid = y_centroid / sum;
		//printf("\n Xmin %d Xmax%d Ymin%d Ymax%d", x_min, x_max, y_min, y_max);
		//printf("\n X_center %d, Y_center %d", x_centroid, y_centroid);

		/*First Moment*/

		for (int i = 0;i < src.rows;i++)
		{
			for (int j = 0;j < src.cols;j++)
			{

				if (src.at<uchar>(i, j) == reg_id)
				{
					X_moment += (i - x_centroid) * (i - x_centroid);
					Y_moment += (j - y_centroid) * (j - y_centroid);
					M11 += (i - x_centroid) * (j - y_centroid);
				}

			}
		}
		mue01 = X_moment / sum;
		mue10 = Y_moment / sum;
		mue11 = M11 / sum;

		alpha = 0.5 * (atan2((2 * mue11), (mue01 - mue10)));
		//printf("ALPHA = %f", alpha);
		x[0] = alpha;
		x[1] = mue01;
		x[2] = mue10;
		x[3] = mue11;
	
	if (save_in_csv > 0) { 
		std::cout << "Enter the Label: ";
		std::cin >> Lab;
		append_image_data_csv(filename, Lab, x, 0); 
	}
	std::vector<char*> filenames;
	std::vector<std::vector<float>> data;
	read_image_data_csv(filename, filenames, data, 0);
	//printf("\ndatasize: %d", data.size());
	float simmi = 10000, simmi_old=10000;
	char best_match[25];
	for (int i = 0; i < data.size()-1;i++)
	{
		float sd = calculateSD(data[i]);
		//printf("\n Standard Deviation %f", sd);

		simmi = distance(data[i],x); /*compare with the last row*/
		if (simmi<simmi_old)
		{
			/*Best match till now*/
			strcpy(best_match, filenames[i]);
			//printf("nex best match %s", best_match);
		}
		simmi_old = simmi;
		
		/*printf("\ndata[i]\n");
		for (int y = 0; y < data[i].size();y++) printf("\t %f", data[i][y]);
		printf("\ndata[i+1]\n");
		for ( int y = 0; y < data[i+1].size();y++) printf("\t %f", data[i+1][y]);
		printf("\n\nsimilarity %f", simmi);*/


	}
	//printf("\n\n The best match found was: %s", best_match);

	/*Insert match text into image*/
	Point text_position(y_centroid, x_centroid);
	//printf("TEXT for %d src cols %d src rows %d", REGIONS[reg], dst.cols, dst.rows);
	int font_size = 1;
	Scalar font_Color(100);//Declaring the color of the font//
	int font_weight = 2;//Declaring the font weight//
	//name[0] = strcat(char(reg),"hi");
	//strcpy(name, string(reg));
	//strcpy(name, "%s");

	putText(src2, best_match, text_position, FONT_HERSHEY_COMPLEX, font_size, font_Color, font_weight);//Putting the text in the matrix//

	return(best_match);
}

int feature_matching()
{
	/*Find and return closest match*/
}


/*
#############Function Description#################
Compute euclidean distance between two vectors
###################################################
*/

float distance(std::vector<float>  x, std::vector<float>  y)
{
	float err = 0;
	float sim;
	//printf("Feature size x= %d and Feature size y = %d", sizeof(x), sizeof(y));
	//for (int j = 0; j < x.size() ;j++) printf("X %d", x[j]);
	//for (int i= sizeof(x)/sizeof(x[0]);i>0;i--)

	for (int i = 0; i < x.size(); i++)
	{
		float diff = (x[i] - y[i]);
		if (diff < 0) diff = -diff;
		err += diff;
		//printf("Distance update: %d", distance);
	}

	sim = (err / x.size());
	// std::cout << "distance = " << distance;
	return sim;
}

