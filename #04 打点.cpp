#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

//Sample points
std::vector<cv::Point2f> sampleContourByArcLength(
    const std::vector<cv::Point>& contour,
    double spacing)
{
    std::vector<cv::Point2f> samples;
    if (contour.size() < 2 || spacing <= 0.0)
        return samples;

    // alculate contour
    double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0)
        return samples;

    //Calculate sample
    int sampleCount = std::max(3,
        static_cast<int>(std::ceil(perimeter / spacing)));
    double step = perimeter / sampleCount;
    std::size_t edgeIndex = 0;
    double edgeStartDistance = 0.0;

    //Equal points
    for (int i = 0; i < sampleCount; ++i)
    {
        double targetDistance = i * step;
        while (edgeIndex < contour.size())
        {
            const cv::Point& p1 = contour[edgeIndex];
            const cv::Point& p2 = contour[(edgeIndex + 1) % contour.size()];
            double segmentLength = cv::norm(p2 - p1);
            if (segmentLength < 1e-6)
            {
                ++edgeIndex;
                continue;
            }
            if (targetDistance <= edgeStartDistance + segmentLength)
            {
                double t = (targetDistance - edgeStartDistance) / segmentLength;
                cv::Point2f point(
                    p1.x + t * (p2.x - p1.x),
                    p1.y + t * (p2.y - p1.y));
                samples.push_back(point);
                break;
            }

            edgeStartDistance += segmentLength;
            ++edgeIndex;
        }
    }
    return samples;
}

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
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    //3.Set the range
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(35,80,80),
        cv::Scalar(85,255,255), mask);

    //4.Get the contour
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(mask.clone(), contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "DETECTED CONTOURS: " << contours.size() << std::endl;

    //5.Sample points
    cv::Mat sampledImage = image.clone();

    const double sampleSpacing = 15.0;
    std::size_t totalSampleCount = 0;

    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        std::vector<cv::Point2f> points =
            sampleContourByArcLength(contours[i], sampleSpacing);

        totalSampleCount += points.size();

        for (std::size_t j = 0; j < points.size(); ++j)
        {
            cv::Point pixel(
                static_cast<int>(std::round(points[j].x)),
                static_cast<int>(std::round(points[j].y))
            );

            cv::circle(sampledImage, pixel, 2,
                cv::Scalar(0,0,255), -1);
        }
    }

    std::cout << "SAMPLED POINTS: " << totalSampleCount << std::endl;

    //6.Show
    cv::imshow("Original", image);
    cv::imshow("Green Mask", mask);
    cv::imshow("Sampled Contour", sampledImage);

    //7.Save
    cv::imwrite("green_mask.png", mask);
    cv::imwrite("sampled_contour.png", sampledImage);

    std::cout << "Completed" << std::endl;
    cv::waitKey(0);
    return 0;
}
