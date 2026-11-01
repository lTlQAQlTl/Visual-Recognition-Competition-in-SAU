#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>

//1.Sample points
std::vector<cv::Point2f> sampleContourByArcLength(
    const std::vector<cv::Point>& contour, double spacing)
{
    std::vector<cv::Point2f> samples;
    if (contour.size() < 2 || spacing <= 0.0)
        return samples;
    double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0) return samples;
    int sampleCount = std::max(3, static_cast<int>(std::ceil(perimeter / spacing)));
    double step = perimeter / sampleCount;
    std::size_t edgeIndex = 0;
    double edgeStartDistance = 0.0;
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
                samples.push_back(cv::Point2f(
                    p1.x + t * (p2.x - p1.x),
                    p1.y + t * (p2.y - p1.y)
                ));
                break;
            }
            edgeStartDistance += segmentLength;
            ++edgeIndex;
        }
    }
    return samples;
}

//2.Calculate circle
bool circleFromThreePoints(
    const cv::Point2f& p1,
    const cv::Point2f& p2,
    const cv::Point2f& p3,
    cv::Point2f& center,
    float& radius)
{
    double x1 = p1.x, y1 = p1.y;
    double x2 = p2.x, y2 = p2.y;
    double x3 = p3.x, y3 = p3.y;
    double d = 2.0 * (
        x1 * (y2 - y3) +
        x2 * (y3 - y1) +
        x3 * (y1 - y2)
    );
    if (std::abs(d) < 1e-6) return false;
    double s1 = x1 * x1 + y1 * y1;
    double s2 = x2 * x2 + y2 * y2;
    double s3 = x3 * x3 + y3 * y3;
    double cx = (s1 * (y2 - y3) +
                 s2 * (y3 - y1) +
                 s3 * (y1 - y2)) / d;
    double cy = (s1 * (x3 - x2) +
                 s2 * (x1 - x3) +
                 s3 * (x2 - x1)) / d;
    if (!std::isfinite(cx) || !std::isfinite(cy))
        return false;
    center = cv::Point2f(cx, cy);
    radius = static_cast<float>(cv::norm(p1 - center));
    return std::isfinite(radius);
}

//3.RANSAC circle fitting
struct CircleModel
{
    cv::Point2f center;
    float radius = 0.0f;
    std::vector<int> inliers;
    double meanError = 0.0;
    bool valid = false;
};
CircleModel fitCircleRANSAC(
    const std::vector<cv::Point2f>& points,
    int iterations,
    double threshold,
    double maxRadius)
{
    CircleModel best;
    if (points.size() < 3) return best;
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, points.size() - 1);
    std::size_t minInliers = std::min<std::size_t>(6, points.size());
    for (int k = 0; k < iterations; ++k)
    {
        // Randomly select three different points
        std::size_t i1 = dist(rng);
        std::size_t i2, i3;
        do { i2 = dist(rng); } while (i2 == i1);
        do { i3 = dist(rng); } while (i3 == i1 || i3 == i2);
        cv::Point2f center;
        float radius;
        if (!circleFromThreePoints(points[i1], points[i2], points[i3], center, radius))
            continue;
        if (radius < 1.0f || radius > maxRadius)
            continue;

        //Evaluate all sampled points
        std::vector<int> inliers;
        double errorSum = 0.0;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            double distance = cv::norm(points[i] - center);
            double error = std::abs(distance - radius);

            if (error <= threshold)
            {
                inliers.push_back(static_cast<int>(i));
                errorSum += error;
            }
        }
        if (inliers.size() < minInliers)
            continue;
        double meanError = errorSum / inliers.size();

        // Keep the best
        if (inliers.size() > best.inliers.size() ||
            (inliers.size() == best.inliers.size() &&
             meanError < best.meanError))
        {
            best.center = center;
            best.radius = radius;
            best.inliers = inliers;
            best.meanError = meanError;
            best.valid = true;
        }
    }
    return best;
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

    //5.Sample and fit circles
    cv::Mat result = image.clone();
    const double sampleSpacing = 15.0;
    const double errorThreshold = 3.0;
    const int ransacIterations = 1000;
    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        std::vector<cv::Point2f> points =
            sampleContourByArcLength(contours[i], sampleSpacing);
        std::cout << "\nCONTOUR " << i
                  << " POINTS: " << points.size() << std::endl;

        // Draw sampled points
        for (const auto& point : points)
        {
            cv::circle(result,
                cv::Point(cvRound(point.x), cvRound(point.y)),
                2, cv::Scalar(0,0,255), -1);
        }

        // Fit circle
        CircleModel model = fitCircleRANSAC(points,ransacIterations,errorThreshold,
            std::max(image.cols, image.rows));
        if (!model.valid)
        {
            std::cout << "NO VALID CIRCLE" << std::endl;
            continue;
        }
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "CENTER: (" << model.center.x
                  << ", " << model.center.y << ")" << std::endl;
        std::cout << "RADIUS: " << model.radius << std::endl;
        std::cout << "INLIERS: " << model.inliers.size()
                  << "/" << points.size() << std::endl;
        std::cout << "MEAN ERROR: " << model.meanError << " px" << std::endl;

        // Draw fitted circle
        cv::Point center(cvRound(model.center.x),cvRound(model.center.y));
        cv::circle(result, center, cvRound(model.radius),
            cv::Scalar(255,0,0), 2);
        cv::drawMarker(result, center,
            cv::Scalar(0,255,255), cv::MARKER_CROSS, 15, 2);
    }

    //6.Show
    cv::imshow("Original", image);
    cv::imshow("Green Mask", mask);
    cv::imshow("RANSAC Result", result);

    //7.Save
    cv::imwrite("green_mask.png", mask);
    cv::imwrite("ransac_result.png", result);
    std::cout << "\nCompleted" << std::endl;

    cv::waitKey(0);
    return 0;
}
