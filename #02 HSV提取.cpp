#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    //1.读取图片
    cv::Mat image = cv::imread("test_output.png");
    if (image.empty())
    {
        std::cout << "Failed" << std::endl;
        return 1;
    }

    //2.转换HSV
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    //3.设置绿色阈值
    cv::Mat mask;
    cv::inRange(hsv,
        cv::Scalar(35, 80, 80),
        cv::Scalar(85, 255, 255),
        mask);

    // 4.显示
    cv::imshow("Original Image", image);
    cv::imshow("Green Mask", mask);

    // 5.保存
    cv::imwrite("green_mask.png", mask);

    std::cout << "Green detection completed!" << std::endl;
    cv::waitKey(0);
    return 0;
}
