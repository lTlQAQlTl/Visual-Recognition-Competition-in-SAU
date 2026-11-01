#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

//1.Sample points along contour
std::vector<cv::Point2f> sampleContourByArcLength(
    const std::vector<cv::Point>& contour,
    double spacing)
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
                samples.push_back(cv::Point2f(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)));
                break;
            }
            edgeStartDistance += segmentLength;
            ++edgeIndex;
        }
    }
    return samples;
}

//2.Calculate a circle from three points
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
    double d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    if (std::abs(d) < 1e-6) return false;
    double s1 = x1 * x1 + y1 * y1;
    double s2 = x2 * x2 + y2 * y2;
    double s3 = x3 * x3 + y3 * y3;
    double cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d;
    double cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d;
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
    std::size_t minInliers = 8;
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
        if (radius < 5.0f || radius > maxRadius)
            continue;

        // Evaluate all sampled points
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

        // Reject circles supported only by a very short arc
        bool angleBins[12] = {};
        for (int index : inliers)
        {
            double angle = std::atan2(points[index].y - center.y, points[index].x - center.x);
            if (angle < 0.0) angle += 2.0 * CV_PI;
            int bin = std::min(11,int(angle / (2.0 * CV_PI) * 12));
            angleBins[bin] = true;
        }
        int occupiedBins = 0;
        for (bool occupied : angleBins)
            if (occupied) ++occupiedBins;
        if (occupiedBins < 6) continue;
        double meanError = errorSum / inliers.size();

        // Keep the best model
        if (inliers.size() > best.inliers.size() || (inliers.size() == best.inliers.size() && meanError < best.meanError))
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

//4.Split colors, then select green groups
cv::Mat getGreenMask(
    const cv::Mat& image,
    cv::Mat& groups)
{
    cv::Mat samples(image.rows * image.cols,3,CV_32F);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(y,x);
            float sum = color[0] + color[1] + color[2] + 1.0f;
            int index = y * image.cols + x;

            // B / sum, G / sum, R / sum
            samples.at<float>(index,0) = color[0] / sum;
            samples.at<float>(index,1) = color[1] / sum;
            samples.at<float>(index,2) = color[2] / sum;
        }
    }
    cv::Mat labels;
    cv::Mat centers;
    int groupCount = std::min(5,samples.rows);
    cv::kmeans(samples, groupCount, labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,30,0.001),
        3,
        cv::KMEANS_PP_CENTERS,
        centers);
    std::vector<bool> isGreen(groupCount,false);
    for (int i = 0; i < groupCount; ++i)
    {
        float b = centers.at<float>(i,0);
        float g = centers.at<float>(i,1);
        float r = centers.at<float>(i,2);
        isGreen[i] = g > r + 0.005f && g > b + 0.02f;
    }
    cv::Mat mask = cv::Mat::zeros(image.size(),CV_8U);
    groups = cv::Mat::zeros(image.size(),CV_8UC3);
    const cv::Vec3b palette[5] = {
        cv::Vec3b(70,70,230), cv::Vec3b(70,220,70), cv::Vec3b(230,100,70), cv::Vec3b(50,220,220), cv::Vec3b(220,70,220)
    };
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            int index = y * image.cols + x;
            int label = labels.at<int>(index,0);
            cv::Vec3b color = image.at<cv::Vec3b>(y,x);
            int brightness = std::max({ int(color[0]),int(color[1]),int(color[2]) });

            // Very dark pixels have unreliable color ratios
            if (brightness < 30) continue;
            groups.at<cv::Vec3b>(y,x) = palette[label];
            if (isGreen[label])
                mask.at<uchar>(y,x) = 255;
        }
    }
    cv::medianBlur(mask,mask,3);
    return mask;
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

    //2.Split colors and get green mask
    cv::setRNGSeed(42);
    cv::Mat groups;
    cv::Mat mask = getGreenMask(image,groups);

    //3.Get the contour
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask.clone(), contours, hierarchy,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);
    std::cout << "DETECTED CONTOURS: " << contours.size() << std::endl;

    //4.Set fitting parameters
    const double sampleSpacing = 3.0;
    const double errorThreshold = 2.0;
    const int ransacIterations = 1000;
    const int maxCirclesPerContour = 30;
    const bool useSizeFilter = true;
    const double minRadiusRatio = 0.60;
    const double maxRadiusRatio = 1.40;
    std::vector<CircleModel> circles;

    //5.Fit several circles from each contour
    for (const auto& contour : contours)
    {
        std::vector<cv::Point2f> points = sampleContourByArcLength(contour,sampleSpacing);
        for (int i = 0; i < maxCirclesPerContour; ++i)
        {
            if (points.size() < 8) break;
            CircleModel model = fitCircleRANSAC(points, ransacIterations, errorThreshold,
                std::max(image.cols,image.rows));
            if (!model.valid) break;
            bool duplicate = false;
            for (const auto& old : circles)
            {
                double smallerRadius = std::min(old.radius,model.radius);
                if (cv::norm(old.center - model.center) < 0.30 * smallerRadius && std::abs(old.radius - model.radius) < 0.25 * smallerRadius)
                {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate)
                circles.push_back(model);

            // Remove only supporting points, not the whole disk
            std::vector<bool> used(points.size(),false);
            for (int index : model.inliers)
                used[index] = true;
            std::vector<cv::Point2f> remaining;
            for (std::size_t j = 0; j < points.size(); ++j)
            {
                if (!used[j])
                    remaining.push_back(points[j]);
            }
            points.swap(remaining);
        }
    }

    //6.Remove unusual sizes using the median radius
    std::vector<CircleModel> kept;
    double medianRadius = 0.0;

    // Too few candidates cannot give a reliable size reference
    if (useSizeFilter && circles.size() >= 5)
    {
        std::vector<float> radii;
        for (const auto& model : circles)
            radii.push_back(model.radius);
        std::sort(radii.begin(),radii.end());
        std::size_t middle = radii.size() / 2;
        medianRadius = radii[middle];
        if (radii.size() % 2 == 0)
            medianRadius = (radii[middle - 1] + radii[middle]) / 2.0;
    }
    for (const auto& model : circles)
    {
        if (medianRadius > 0.0 && (model.radius < minRadiusRatio * medianRadius || model.radius > maxRadiusRatio * medianRadius))
            continue;
        kept.push_back(model);
    }

    //7.Draw circles and boxes
    cv::Mat result = image.clone();
    cv::Rect imageArea(0,0,image.cols,image.rows);
    for (std::size_t i = 0; i < kept.size(); ++i)
    {
        const CircleModel& model = kept[i];
        cv::Point center(cvRound(model.center.x),cvRound(model.center.y));
        int radius = cvRound(model.radius);
        cv::circle(result, center, radius,
            cv::Scalar(255,0,0),
            2);
        cv::Rect box(center.x - radius,center.y - radius, 2 * radius + 1,2 * radius + 1);
        box &= imageArea;
        if (box.area() > 0)
            cv::rectangle(result, box,
                cv::Scalar(0,0,255),
                1);
        cv::putText(result,
            std::to_string(i + 1),
            center,
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0,0,255),
            1);
    }
    cv::putText(result, "COUNT: " + std::to_string(kept.size()),
        cv::Point(10,30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0,0,255),
        2);
    std::cout << "CANDIDATES: " << circles.size() << std::endl;
    std::cout << "MEDIAN RADIUS: " << medianRadius << std::endl;
    std::cout << "GREEN BALLS: " << kept.size() << std::endl;

    //8.Show
    cv::imshow("Original",image);
    cv::imshow("Color Groups",groups);
    cv::imshow("Green Mask",mask);
    cv::imshow("RANSAC Result",result);

    //9.Save
    cv::imwrite("color_groups.png",groups);
    cv::imwrite("green_mask.png",mask);
    cv::imwrite("ransac_result.png",result);
    std::cout << "Completed" << std::endl;
    cv::waitKey(0);
    return 0;
}
