#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    //创建图片
    cv::Mat image(600,800,CV_8UC3,
        cv::Scalar(255, 255, 255));

    //画图
    cv::circle(image,cv::Point(400, 300),100,
        cv::Scalar(0, 255, 0),
        -1);

    //显示
    cv::imshow("My First OpenCV", image);

    //保存
    cv::imwrite("test_output.png", image);
    
    std::cout << "Image generated successfully!" << std::endl;
    cv::waitKey(0);
    return 0;
}
