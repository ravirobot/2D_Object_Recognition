/*
* CS5330: Assignment 3
Author: RAVI MAHESHWARI
NUID:002104786
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "functions.h"
//#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

float M00, M01, M10, M11;
int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;
    cv::Mat img1,img2,img3,img4, img5;
 
    int key1 = 0;
    int time = 0;
    cv::Mat result_frame;
    std::vector<float>  x = {1.2, 3.2};
    cv::Mat img = cv::imread("C:/Northeastern/PRCV/visual C/ObjectRecognition/ObjectRecognition/objects_database/pen1.jpg");

    cv::imshow("Original image",img);
    blur5x5(img,img1);
    binarization(img1, img2,450);
    cv::imshow("Binary image", img2);
    grassfire(img2, img3);
    // TODO Implement growing and shrinking
    cv::imshow("Distance Transformed", img3);
    segmentation(img2, img4);
    viewSegments(img4, result_frame, -1);
    cv::imshow("Distance Transformed", img3);
    cv::imshow("Segments", result_frame);
    /*wont work at binarization input is 3 channel 
    binarization(img3, img4, 5);
    cv::imshow("Distance Binary Transformed", img4);  
    wont work at binarization input is 3 channel*/

    moments(img4, result_frame);
    //TODO : Show boxes on original image
    cv::imshow("Bounding Box", result_frame);
    waitKey(0);
    /*
    ###################################
    FEATURE EXTRACTION AND STORAGE
    ###################################
    */
    int store_in_csv = 1;
    extract_features(img4, result_frame, store_in_csv);
//    append_image_data_csv(filename, Lab, x, 0);

    /*################################
    * VIDEO
    ################################*/

    //int numberOfDevices = 0;
    //int noError = 0;

    //while (numberOfDevices <10)
    //{
    //    try
    //    {
    //        // Check if camera is available.
    //        VideoCapture videoCapture(numberOfDevices); // Will crash if not available, hence try/catch.

    //        // ...
    //    }
    //    catch (...)
    //    {
    //        noError++;
    //    }

    //    // If above call worked, we have found another camera.
    //    
    //    printf("num%d", numberOfDevices);
    //    numberOfDevices++;
    //}

    /*Capture from cell: 
    do not START, but open both DroidCam clients*/
    capdev = new cv::VideoCapture("http://10.0.0.177:4747/mjpegfeed?640x480"); 

    /*Capture from laptop*/
     //capdev = new cv::VideoCapture(0); 
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::Mat frame;
    int enable_video = 0;
    for (;enable_video > 0 ;) {
        time++;
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        frame.copyTo(result_frame);
        cv::imshow("Original  Video", frame);
        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        //key = cv::waitKey(10);
        if (key == 's') {
            blur5x5(frame, img1);
            binarization(img1, img2, 450);
            grassfire(img2, img3);
            segmentation(img2, img4);
            viewSegments(img4, result_frame, -1);
            moments(img4, result_frame);        
            store_in_csv = 1;
            extract_features(img4, result_frame, store_in_csv);
        }

        if (key == 'g') key1 = (key1 + 1) % 2;
        if (key1 == 1)
        {
            //cv::cvtColor(frame, result_frame, cv::COLOR_BGR2GRAY);
        }
       
        blur5x5(frame, img1);
        binarization(img1, img2, 300);
        //
        //cv::imshow("Binary image", img2);
        //grassfire(img2, img3);
        segmentation(img2, img4);
        moments(img4, result_frame);
        //cv::imshow("Distance Transformed", img3);
        extract_features(img4, result_frame, 0);
        cv::imshow("Resultant  Video", result_frame);
    }

    delete capdev;
    return(0);
}

