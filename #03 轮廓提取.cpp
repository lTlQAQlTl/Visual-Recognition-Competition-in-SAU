#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main()
{
    //1.Read the pic
    cv::Mat image = cv::imread("test_output.png");
    if (image.empty())
    {
        std::cout << "Failed" << std::endl;
        return 1;
    }

    //2.Trans to HSV
    cv::Mat hsv;
    cv::cvtColor(image,hsv,
        cv::COLOR_BGR2HSV);

    //3.Set the range
    cv::Mat mask;
    cv::inRange(hsv,
        cv::Scalar(35,80,80),
        cv::Scalar(85,255,255),
        mask);

    //4.Get the contour
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask.clone(),contours,hierarchy,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);
    std::cout << "DETECTED CONTOURS:" << contours.size() <<std::endl;

    //5.Show
    cv::imshow("Original",image);
    cv::imshow("Green Mask",mask);

    //6.Save
    cv::imwrite("green_mask.png",mask);

    std::cout << "Completed" << std::endl;
    cv::waitKey(0);
    return 0;
}