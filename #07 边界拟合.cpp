#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>

//0.Parameters
struct Params
{
    // Ball color
    int hueMin = 28;
    int hueMax = 50;
    int chromaMin = 20;
    int chromaMax = 95;
    int brightnessMin = 45;

    // Border detection
    double gradientMin = 0.0035;
    int sideOffset = 3;

    // Circle fitting
    double fitError = 3.0;
    float radiusMin = 6.0f;
    double minRadiusFraction = 1.0 / 70;
    int maxIterations = 4000;
    int refineIterations = 30000;
    double refineWindow = 2.2;

    // Circle scoring
    int perimeterSamples = 72;
    double edgeSearch = 3.0;
    double minEdgeSupport = 0.30;
    double minInsideColor = 0.60;

    // Selection
    double overlapRatio = 0.75;
    double referenceEdgeSupport = 0.60;
    double minRadiusRatio = 0.80;
    double maxRadiusRatio = 1.25;

    // Final check
    double finalInsideColor = 0.85;
    double maxInnerChroma = 70.0;
    double maxUnexplainedOutside = 0.30;
    double trustedEdgeSupport = 0.50;
};
struct CircleModel
{
    cv::Point2f center;
    float radius = 0.0f;
    double edgeSupport = 0.0;
    double insideColor = 0.0;
};

//1.Greenness map
cv::Mat computeGreenness(
    const cv::Mat& image)
{
    cv::Mat greenness(image.size(), CV_32F);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            float sum = c[0] + c[1] + c[2] + 1.0f;
            greenness.at<float>(y, x) = (c[1] - 0.5f * (c[0] + c[2])) / sum;
        }
    }

    // Smooth JPEG color blocks
    cv::GaussianBlur(greenness, greenness,
        cv::Size(9, 9),
        2.0);
    return greenness;
}

//2.Ball color check for every pixel
cv::Mat computeBallColor(
    const cv::Mat& image,
    const Params& p)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv,
        cv::COLOR_BGR2HSV);
    cv::Mat ballColor = cv::Mat::zeros(image.size(), CV_8U);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            int maxValue = std::max({ int(c[0]), int(c[1]), int(c[2]) });
            int minValue = std::min({ int(c[0]), int(c[1]), int(c[2]) });
            int chroma = maxValue - minValue;
            int hue = hsv.at<cv::Vec3b>(y, x)[0];
            if (hue >= p.hueMin && hue <= p.hueMax && chroma >= p.chromaMin && chroma <= p.chromaMax && maxValue >= p.brightnessMin)
                ballColor.at<uchar>(y, x) = 255;
        }
    }
    return ballColor;
}

//3.Find border points
std::vector<cv::Point2f> findBorderPoints(
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& ballColor,
    const Params& p,
    cv::Mat& borderMap)
{
    std::vector<cv::Point2f> points;
    borderMap = cv::Mat::zeros(gradX.size(), CV_8U);
    int rows = gradX.rows, cols = gradX.cols;
    for (int y = 1; y < rows - 1; ++y)
    {
        for (int x = 1; x < cols - 1; ++x)
        {
            float gx = gradX.at<float>(y, x);
            float gy = gradY.at<float>(y, x);
            float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < p.gradientMin) continue;

            // Keep the local maximum only
            float dx = gx / mag, dy = gy / mag;
            auto magAt = [&](float fx, float fy)
            {
                int ix = std::clamp(cvRound(fx), 0, cols - 1);
                int iy = std::clamp(cvRound(fy), 0, rows - 1);
                float a = gradX.at<float>(iy, ix), b = gradY.at<float>(iy, ix);
                return std::sqrt(a * a + b * b);
            };
            if (mag < magAt(x + dx, y + dy) || mag < magAt(x - dx, y - dy))
                continue;

            // Greener side must be ball color
            int ix = cvRound(x + p.sideOffset * dx);
            int iy = cvRound(y + p.sideOffset * dy);
            if (ix < 0 || iy < 0 || ix >= cols || iy >= rows) continue;
            if (!ballColor.at<uchar>(iy, ix)) continue;
            points.push_back(cv::Point2f(float(x), float(y)));
            borderMap.at<uchar>(y, x) = 255;
        }
    }
    return points;
}

//4.Group border points
std::vector<std::vector<cv::Point2f>> groupBorderPoints(
    const cv::Mat& borderMap,
    int minPoints)
{
    // Join broken border lines
    cv::Mat bridged, labels;
    cv::dilate(borderMap, bridged,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    int count = cv::connectedComponents(bridged, labels, 8, CV_32S);
    std::vector<std::vector<cv::Point2f>> groups(count);
    for (int y = 0; y < borderMap.rows; ++y)
        for (int x = 0; x < borderMap.cols; ++x)
            if (borderMap.at<uchar>(y, x))
                groups[labels.at<int>(y, x)].push_back(cv::Point2f(float(x), float(y)));
    std::vector<std::vector<cv::Point2f>> kept;
    for (auto& g : groups)
        if (static_cast<int>(g.size()) >= minPoints) kept.push_back(std::move(g));
    return kept;
}

//5.Circle from three points
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

//6.Least-squares circle fitting
bool fitCircleLeastSquares(
    const std::vector<cv::Point2f>& pts,
    cv::Point2f& center,
    float& radius)
{
    if (pts.size() < 3) return false;
    double mx = 0, my = 0;
    for (const auto& q : pts) { mx += q.x; my += q.y; }
    mx /= pts.size(); my /= pts.size();
    cv::Mat A(int(pts.size()), 3, CV_64F), b(int(pts.size()), 1, CV_64F);
    for (int i = 0; i < int(pts.size()); ++i)
    {
        double x = pts[i].x - mx, y = pts[i].y - my;
        A.at<double>(i, 0) = x;
        A.at<double>(i, 1) = y;
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = -(x * x + y * y);
    }
    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) return false;
    double D = sol.at<double>(0), E = sol.at<double>(1);
    double F = sol.at<double>(2);
    double r2 = D * D / 4 + E * E / 4 - F;
    if (r2 <= 0) return false;
    center = cv::Point2f(float(mx - D / 2), float(my - E / 2));
    radius = float(std::sqrt(r2));
    return true;
}

//7.Score a circle
void scoreCircle(
    CircleModel& c,
    const cv::Mat& nearBorder,
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& ballColor,
    const Params& p)
{
    int rows = ballColor.rows, cols = ballColor.cols;
    int supported = 0, total = 0;
    for (int k = 0; k < p.perimeterSamples; ++k)
    {
        double a = 2.0 * CV_PI * k / p.perimeterSamples;
        double nx = std::cos(a), ny = std::sin(a);
        int x = cvRound(c.center.x + c.radius * nx);
        int y = cvRound(c.center.y + c.radius * ny);
        ++total;
        if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
        if (!nearBorder.at<uchar>(y, x)) continue;
        float gx = gradX.at<float>(y, x), gy = gradY.at<float>(y, x);
        float mag = std::sqrt(gx * gx + gy * gy);
        if (mag < 1e-6f) continue;

        // Greener toward the center
        double cosine = -(gx * nx + gy * ny) / mag;
        if (cosine > 0.5) ++supported;
    }
    c.edgeSupport = double(supported) / total;
    int inside = 0, checked = 0;
    for (double rr : { 0.0, 0.35, 0.7 })
    {
        int n = (rr == 0.0) ? 1 : 12;
        for (int k = 0; k < n; ++k)
        {
            double a = 2.0 * CV_PI * k / n;
            int x = cvRound(c.center.x + rr * c.radius * std::cos(a));
            int y = cvRound(c.center.y + rr * c.radius * std::sin(a));
            ++checked;
            if (x >= 0 && y >= 0 && x < cols && y < rows && ballColor.at<uchar>(y, x))
                ++inside;
        }
    }
    c.insideColor = double(inside) / checked;
}

//8.Propose circles with RANSAC
std::vector<CircleModel> proposeCircles(
    const std::vector<cv::Point2f>& pts,
    float radiusLow,
    float radiusHigh,
    double window,
    int iterationsPerPoint,
    int maxIterations,
    const cv::Mat& nearBorder,
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& ballColor,
    const Params& p,
    std::mt19937& rng)
{
    std::vector<CircleModel> proposals;
    if (pts.size() < 8) return proposals;
    std::uniform_int_distribution<std::size_t> pick(0, pts.size() - 1);
    int iterations = std::min(maxIterations, 300 + iterationsPerPoint * int(pts.size()));
    std::vector<int> near;
    for (int it = 0; it < iterations; ++it)
    {
        const cv::Point2f& a = pts[pick(rng)];
        near.clear();
        for (int i = 0; i < int(pts.size()); ++i)
        {
            double d = cv::norm(pts[i] - a);
            if (d >= 4.0 && d <= window) near.push_back(i);
        }
        if (near.size() < 2) continue;
        std::uniform_int_distribution<std::size_t> pickNear(0, near.size() - 1);
        const cv::Point2f& b = pts[near[pickNear(rng)]];
        const cv::Point2f& c = pts[near[pickNear(rng)]];
        if (cv::norm(b - c) < 4.0) continue;
        CircleModel m;
        if (!circleFromThreePoints(a, b, c, m.center, m.radius)) continue;
        if (m.radius < radiusLow || m.radius > radiusHigh) continue;

        // Inliers, then regression
        std::vector<cv::Point2f> inliers;
        for (const auto& q : pts)
            if (std::abs(cv::norm(q - m.center) - m.radius) <= p.fitError)
                inliers.push_back(q);

        // Enough inliers
        if (inliers.size() < std::max<std::size_t>(8, std::size_t(0.25 * 2 * CV_PI * m.radius)))
            continue;
        if (!fitCircleLeastSquares(inliers, m.center, m.radius)) continue;
        if (m.radius < radiusLow || m.radius > radiusHigh) continue;
        scoreCircle(m, nearBorder, gradX, gradY, ballColor, p);
        if (m.edgeSupport >= p.minEdgeSupport && m.insideColor >= p.minInsideColor)
            proposals.push_back(m);
    }
    return proposals;
}

//9.Final check of circles
double innerChroma(
    const CircleModel& c,
    const cv::Mat& image)
{
    double sum = 0.0;
    int count = 0;
    int r = cvRound(0.7 * c.radius);
    for (int dy = -r; dy <= r; dy += 2)
    {
        for (int dx = -r; dx <= r; dx += 2)
        {
            if (dx * dx + dy * dy > r * r) continue;
            int x = cvRound(c.center.x) + dx, y = cvRound(c.center.y) + dy;
            if (x < 0 || y < 0 || x >= image.cols || y >= image.rows) continue;
            cv::Vec3b v = image.at<cv::Vec3b>(y, x);
            sum += std::max({ v[0], v[1], v[2] }) - std::min({ v[0], v[1], v[2] });
            ++count;
        }
    }
    return count ? sum / count : 0.0;
}
double unexplainedOutside(
    std::size_t self,
    const std::vector<CircleModel>& circles,
    const cv::Mat& ballColor)
{
    const CircleModel& c = circles[self];
    int green = 0, total = 0;
    for (double scale : { 1.25, 1.40 })
    {
        for (int k = 0; k < 72; ++k)
        {
            double a = 2.0 * CV_PI * k / 72;
            cv::Point2f q(float(c.center.x + scale * c.radius * std::cos(a)), float(c.center.y + scale * c.radius * std::sin(a)));
            int x = cvRound(q.x), y = cvRound(q.y);
            if (x < 0 || y < 0 || x >= ballColor.cols || y >= ballColor.rows) continue;
            ++total;
            if (!ballColor.at<uchar>(y, x)) continue;
            bool explained = false;
            for (std::size_t j = 0; j < circles.size() && !explained; ++j)
                if (j != self && cv::norm(q - circles[j].center) <= 1.05 * circles[j].radius)
                    explained = true;
            if (!explained) ++green;
        }
    }
    return total ? double(green) / total : 0.0;
}
std::vector<CircleModel> verifyCircles(
    std::vector<CircleModel> circles,
    const cv::Mat& image,
    const cv::Mat& ballColor,
    const Params& p)
{
    std::vector<CircleModel> kept;
    for (const auto& c : circles)
        if (c.insideColor >= p.finalInsideColor && innerChroma(c, image) <= p.maxInnerChroma)
            kept.push_back(c);

    // Remove the worst circle first
    while (true)
    {
        double worst = p.maxUnexplainedOutside;
        int worstIndex = -1;
        for (std::size_t i = 0; i < kept.size(); ++i)
        {
            if (kept[i].edgeSupport >= p.trustedEdgeSupport) continue;
            double u = unexplainedOutside(i, kept, ballColor);
            if (u > worst) { worst = u; worstIndex = int(i); }
        }
        if (worstIndex < 0) break;
        kept.erase(kept.begin() + worstIndex);
    }
    return kept;
}

//10.Select circles
std::vector<CircleModel> selectCircles(
    std::vector<CircleModel> proposals,
    const Params& p,
    float radiusLow,
    float radiusHigh)
{
    // Longer border arc first
    std::sort(proposals.begin(), proposals.end(),
        [](const CircleModel& a, const CircleModel& b) { return a.edgeSupport * a.radius > b.edgeSupport * b.radius; });
    std::vector<CircleModel> chosen;
    for (const auto& m : proposals)
    {
        if (m.radius < radiusLow || m.radius > radiusHigh) continue;
        bool conflict = false;
        for (const auto& old : chosen)
        {
            if (cv::norm(old.center - m.center) < p.overlapRatio * (old.radius + m.radius))
            {
                conflict = true;
                break;
            }
        }
        if (!conflict) chosen.push_back(m);
    }
    return chosen;
}

int main(int argc, char** argv)
{
    //1.Read the pic
    std::string path = (argc > 1) ? argv[1] : "test_output.png";
    bool show = (argc > 2) && std::string(argv[2]) == "--show";
    cv::Mat image = cv::imread(path);
    if (image.empty())
    {
        std::cout << "Failed to read " << path << std::endl;
        return 1;
    }
    Params p;

    //2.Greenness and ball color
    cv::Mat greenness = computeGreenness(image);
    cv::Mat gradX, gradY;
    cv::Sobel(greenness, gradX, CV_32F, 1, 0, 3, 1.0 / 8);
    cv::Sobel(greenness, gradY, CV_32F, 0, 1, 3, 1.0 / 8);
    cv::Mat ballColor = computeBallColor(image, p);

    //3.Get border points
    cv::Mat borderMap;
    std::vector<cv::Point2f> border = findBorderPoints(gradX, gradY, ballColor, p, borderMap);
    cv::Mat nearBorder;
    int k = 2 * int(p.edgeSearch) + 1;
    cv::dilate(borderMap, nearBorder,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    std::cout << "BORDER POINTS: " << border.size() << std::endl;

    //4.Group border points
    std::vector<std::vector<cv::Point2f>> groups = groupBorderPoints(borderMap, 20);
    std::cout << "BORDER GROUPS: " << groups.size() << std::endl;

    //5.First pass fitting
    float radiusMax = std::min(image.cols, image.rows) / 8.0f;
    p.radiusMin = std::max(6.0f, float(std::min(image.cols, image.rows) * p.minRadiusFraction));
    std::mt19937 rng(42);
    std::vector<CircleModel> proposals;
    for (const auto& g : groups)
    {
        auto part = proposeCircles(g, p.radiusMin, radiusMax, 2.0 * radiusMax, 4, p.maxIterations, nearBorder, gradX, gradY, ballColor, p, rng);
        proposals.insert(proposals.end(), part.begin(), part.end());
    }
    std::cout << "PROPOSALS: " << proposals.size() << std::endl;

    //6.Get typical ball size
    std::vector<CircleModel> first = selectCircles(proposals, p, 0.0f, radiusMax);
    std::vector<float> radii;
    for (const auto& m : first)
        if (m.edgeSupport >= p.referenceEdgeSupport) radii.push_back(m.radius);
    if (radii.empty())
        for (const auto& m : first) radii.push_back(m.radius);
    float medianRadius = 0.0f;
    if (!radii.empty())
    {
        std::nth_element(radii.begin(), radii.begin() + radii.size() / 2, radii.end());
        medianRadius = radii[radii.size() / 2];
    }
    //7.Second pass fitting
    float radiusLow = float(p.minRadiusRatio * medianRadius);
    float radiusHigh = float(p.maxRadiusRatio * medianRadius);
    for (const auto& g : groups)
    {
        auto part = proposeCircles(g, radiusLow, radiusHigh, p.refineWindow * medianRadius, 8, p.refineIterations, nearBorder, gradX, gradY, ballColor, p, rng);
        proposals.insert(proposals.end(), part.begin(), part.end());
    }
    std::vector<CircleModel> chosen = selectCircles(proposals, p, radiusLow, radiusHigh);
    std::cout << "CHOSEN: " << chosen.size() << std::endl;

    //8.Final check
    std::vector<CircleModel> balls = verifyCircles(chosen, image, ballColor, p);

    //9.Draw circles and boxes
    cv::Mat result = image.clone();
    cv::Rect imageArea(0, 0, image.cols, image.rows);
    for (std::size_t i = 0; i < balls.size(); ++i)
    {
        cv::Point center(cvRound(balls[i].center.x), cvRound(balls[i].center.y));
        int radius = cvRound(balls[i].radius);
        cv::circle(result, center, radius,
            cv::Scalar(255,0,0),
            2);
        cv::Rect box(center.x - radius, center.y - radius, 2 * radius + 1, 2 * radius + 1);
        box &= imageArea;
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
    cv::putText(result, "COUNT: " + std::to_string(balls.size()),
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0,0,255),
        2);
    std::cout << "MEDIAN RADIUS: " << medianRadius << std::endl;
    std::cout << "GREEN BALLS: " << balls.size() << std::endl;

    //10.Save
    cv::imwrite("border_map.png", borderMap);
    cv::imwrite("ball_color.png", ballColor);
    cv::imwrite("edge_result.png", result);
    if (show)
    {
        cv::imshow("Border", borderMap);
        cv::imshow("Result", result);
        cv::waitKey(0);
    }
    std::cout << "Completed" << std::endl;
    return 0;
}
