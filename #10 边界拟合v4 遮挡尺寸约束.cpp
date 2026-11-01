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

    // Brightness weighting
    int darkLevel = 30;
    int fullLevel = 90;

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
    double minRadiusRatio = 0.85;
    double maxRadiusRatio = 1.20;

    // Final check
    double finalInsideColor = 0.80;
    double maxInnerChroma = 70.0;
    double maxUnexplainedOutside = 0.30;
    double trustedEdgeSupport = 0.50;

    // Occlusion reverse test
    double minReverseSupport = 0.25;
    double maxOccluderInside = 0.20;
    double minOccluderRatio = 0.70;
    double maxOccluderRatio = 1.30;
    double minVisible = 0.50;
    double occludedVisible = 0.90;
    double occludedMinRatio = 0.95;
    double occludedMaxRatio = 1.05;
    double maxUnexplainedTrusted = 0.50;
};
struct CircleModel
{
    cv::Point2f center;
    float radius = 0.0f;
    double edgeSupport = 0.0;
    double insideColor = 0.0;
    double reverseSupport = 0.0;
    double visible = 1.0;
};

//1.Greenness map
cv::Mat computeGreenness(
    const cv::Mat& image,
    const Params& p)
{
    cv::Mat greenness(image.size(), CV_32F);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            float sum = c[0] + c[1] + c[2] + 1.0f;
            float ratio = (c[1] - 0.5f * (c[0] + c[2])) / sum;
            int brightness = std::max({ int(c[0]), int(c[1]), int(c[2]) });
            float weight = std::clamp(float(brightness - p.darkLevel) / float(p.fullLevel - p.darkLevel), 0.0f, 1.0f);
            greenness.at<float>(y, x) = ratio * weight;
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
    const cv::Mat& occluders,
    const Params& p)
{
    int rows = ballColor.rows, cols = ballColor.cols;
    auto hidden = [&](int x, int y)
    {
        return !occluders.empty() && x >= 0 && y >= 0 && x < cols && y < rows && occluders.at<uchar>(y, x);
    };
    int supported = 0, reverse = 0;
    int visibleCount = 0;
    for (int k = 0; k < p.perimeterSamples; ++k)
    {
        double a = 2.0 * CV_PI * k / p.perimeterSamples;
        double nx = std::cos(a), ny = std::sin(a);
        int x = cvRound(c.center.x + c.radius * nx);
        int y = cvRound(c.center.y + c.radius * ny);
        if (hidden(x, y)) continue;
        ++visibleCount;
        if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
        if (!nearBorder.at<uchar>(y, x)) continue;
        float gx = gradX.at<float>(y, x), gy = gradY.at<float>(y, x);
        float mag = std::sqrt(gx * gx + gy * gy);
        if (mag < 1e-6f) continue;

        // Inward: own border, outward: front object
        double cosine = -(gx * nx + gy * ny) / mag;
        if (cosine > 0.5) ++supported;
        else if (cosine < -0.5) ++reverse;
    }
    c.visible = double(visibleCount) / p.perimeterSamples;
    c.edgeSupport = visibleCount ? double(supported) / visibleCount : 0.0;
    c.reverseSupport = double(reverse) / p.perimeterSamples;
    int inside = 0, checked = 0;
    for (double rr : { 0.0, 0.35, 0.7 })
    {
        int n = (rr == 0.0) ? 1 : 12;
        for (int k = 0; k < n; ++k)
        {
            double a = 2.0 * CV_PI * k / n;
            int x = cvRound(c.center.x + rr * c.radius * std::cos(a));
            int y = cvRound(c.center.y + rr * c.radius * std::sin(a));
            if (hidden(x, y)) continue;
            ++checked;
            if (x >= 0 && y >= 0 && x < cols && y < rows && ballColor.at<uchar>(y, x))
                ++inside;
        }
    }
    c.insideColor = checked ? double(inside) / checked : 0.0;
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
    std::mt19937& rng,
    std::vector<CircleModel>* reverseCircles = nullptr)
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
        scoreCircle(m, nearBorder, gradX, gradY, ballColor,
            cv::Mat(),
            p);
        if (m.edgeSupport >= p.minEdgeSupport && m.insideColor >= p.minInsideColor)
            proposals.push_back(m);
        else if (reverseCircles && m.reverseSupport >= p.minReverseSupport && m.insideColor <= p.maxOccluderInside)
            reverseCircles->push_back(m);
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
struct DroppedCircle
{
    CircleModel circle;
    std::string reason;
};
std::vector<CircleModel> verifyCircles(
    std::vector<CircleModel> circles,
    const cv::Mat& image,
    const cv::Mat& ballColor,
    const Params& p,
    std::vector<DroppedCircle>& dropped)
{
    std::vector<CircleModel> kept;
    for (const auto& c : circles)
    {
        if (c.insideColor < p.finalInsideColor)
            dropped.push_back({ c, "inside" });
        else if (innerChroma(c, image) > p.maxInnerChroma)
            dropped.push_back({ c, "chroma" });
        else
            kept.push_back(c);
    }

    // Remove the worst circle first
    while (true)
    {
        double worst = 0.0;
        int worstIndex = -1;
        for (std::size_t i = 0; i < kept.size(); ++i)
        {
            double limit = (kept[i].edgeSupport >= p.trustedEdgeSupport) ? p.maxUnexplainedTrusted : p.maxUnexplainedOutside;
            double u = unexplainedOutside(i, kept, ballColor);
            if (u > limit && u - limit > worst) { worst = u - limit; worstIndex = int(i); }
        }
        if (worstIndex < 0) break;
        dropped.push_back({ kept[worstIndex], "outside" });
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

//11.Select occluders
std::vector<CircleModel> selectOccluders(
    std::vector<CircleModel> reverse,
    const Params& p,
    float radiusLow,
    float radiusHigh)
{
    std::sort(reverse.begin(), reverse.end(),
        [](const CircleModel& a, const CircleModel& b) { return a.reverseSupport * a.radius > b.reverseSupport * b.radius; });
    std::vector<CircleModel> chosen;
    for (const auto& m : reverse)
    {
        if (m.radius < radiusLow || m.radius > radiusHigh) continue;
        bool conflict = false;
        for (const auto& old : chosen)
            if (cv::norm(old.center - m.center) < p.overlapRatio * (old.radius + m.radius))
                conflict = true;
        if (!conflict) chosen.push_back(m);
    }
    return chosen;
}

// Border points no green circle explains
cv::Mat leftoverBorder(
    const cv::Mat& borderMap,
    const std::vector<CircleModel>& green,
    const Params& p)
{
    cv::Mat explained = cv::Mat::zeros(borderMap.size(), CV_8U);
    int thickness = 2 * int(std::ceil(p.fitError)) + 1;
    for (const auto& g : green)
        cv::circle(explained,
            cv::Point(cvRound(g.center.x), cvRound(g.center.y)),
            cvRound(g.radius),
            cv::Scalar(255),
            thickness);
    cv::Mat leftover;
    cv::bitwise_and(borderMap, ~explained, leftover);
    return leftover;
}
cv::Mat occluderMask(
    const std::vector<CircleModel>& occluders,
    cv::Size size)
{
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    for (const auto& o : occluders)
        cv::circle(mask,
            cv::Point(cvRound(o.center.x), cvRound(o.center.y)),
            cvRound(o.radius),
            cv::Scalar(255),
            cv::FILLED);
    return mask;
}

//12.Save process images
cv::Mat dimmed(
    const cv::Mat& image)
{
    cv::Mat out;
    image.convertTo(out, -1, 0.45, 0);
    return out;
}
void drawCircle(
    cv::Mat& canvas,
    const CircleModel& c,
    const cv::Scalar& color,
    int thickness)
{
    cv::circle(canvas,
        cv::Point(cvRound(c.center.x), cvRound(c.center.y)),
        cvRound(c.radius), color, thickness);
}
void saveProcessImages(
    const cv::Mat& image,
    const cv::Mat& greenness,
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& ballColor,
    const cv::Mat& borderMap,
    const std::vector<std::vector<cv::Point2f>>& groups,
    const cv::Mat& leftover,
    const std::vector<CircleModel>& occluders,
    const std::vector<CircleModel>& chosen,
    const std::vector<DroppedCircle>& dropped,
    const cv::Mat& result,
    bool show)
{
    std::vector<std::pair<std::string, cv::Mat>> steps;

    // 1.Greenness
    cv::Mat green8;
    greenness.convertTo(green8, CV_8U, 1500.0, 60.0);
    steps.push_back({ "1_greenness", green8 });

    // 2.Ball color
    cv::Mat colorView = dimmed(image);
    colorView.setTo(cv::Scalar(255,0,255),
        ballColor);
    steps.push_back({ "2_ball_color", colorView });

    // 3.Change rate
    cv::Mat magnitude, rate8;
    cv::magnitude(gradX, gradY, magnitude);
    magnitude.convertTo(rate8, CV_8U, 20000.0);
    steps.push_back({ "3_change_rate", rate8 });

    // 4.Border points
    cv::Mat borderView = dimmed(image);
    borderView.setTo(cv::Scalar(0,0,255),
        borderMap);
    steps.push_back({ "4_border_points", borderView });

    // 5.Border groups
    cv::Mat groupView = dimmed(image);
    cv::RNG colorRng(7);
    for (const auto& g : groups)
    {
        cv::Vec3b color(uchar(colorRng.uniform(80, 256)), uchar(colorRng.uniform(80, 256)), uchar(colorRng.uniform(80, 256)));
        for (const auto& q : g)
            groupView.at<cv::Vec3b>(cvRound(q.y), cvRound(q.x)) = color;
    }
    steps.push_back({ "5_border_groups", groupView });

    // 6.Occluders
    cv::Mat occluderView = dimmed(image);
    occluderView.setTo(cv::Scalar(0,0,255),
        leftover);
    for (const auto& o : occluders) drawCircle(occluderView, o, cv::Scalar(0,255,255), 2);
    steps.push_back({ "6_occluders", occluderView });

    // 7.Candidates
    cv::Mat candidateView = image.clone();
    for (const auto& c : chosen) drawCircle(candidateView, c, cv::Scalar(255,0,0), 2);
    for (const auto& d : dropped)
    {
        drawCircle(candidateView, d.circle,
            cv::Scalar(0,140,255),
            2);
        cv::putText(candidateView, d.reason,
            cv::Point(cvRound(d.circle.center.x - d.circle.radius), cvRound(d.circle.center.y)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0,140,255),
            2);
    }
    steps.push_back({ "7_candidates", candidateView });

    // 8.Result
    steps.push_back({ "8_result", result });
    for (const auto& s : steps)
    {
        cv::imwrite(s.first + ".png", s.second);
        if (show) cv::imshow(s.first, s.second);
    }
    if (show) cv::waitKey(0);
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
    cv::Mat greenness = computeGreenness(image, p);
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
    float occluderLow = float(p.minOccluderRatio * medianRadius);
    float occluderHigh = float(p.maxOccluderRatio * medianRadius);
    std::vector<CircleModel> reverse;
    for (const auto& g : groups)
    {
        auto part = proposeCircles(g,
            std::min(radiusLow, occluderLow),
            std::max(radiusHigh, occluderHigh),
            p.refineWindow * medianRadius, 8, p.refineIterations, nearBorder, gradX, gradY, ballColor, p, rng, &reverse);
        proposals.insert(proposals.end(), part.begin(), part.end());
    }

    //8.Reverse test
    std::vector<CircleModel> plainChosen = selectCircles(proposals, p, radiusLow, radiusHigh);
    cv::Mat leftover = leftoverBorder(borderMap, plainChosen, p);
    cv::Mat nearLeftover;
    cv::dilate(leftover, nearLeftover,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    std::vector<CircleModel> leftoverReverse;
    for (CircleModel m : reverse)
    {
        scoreCircle(m, nearLeftover, gradX, gradY, ballColor,
            cv::Mat(),
            p);
        if (m.reverseSupport >= p.minReverseSupport) leftoverReverse.push_back(m);
    }
    std::vector<CircleModel> occluders = selectOccluders(leftoverReverse, p, occluderLow, occluderHigh);
    cv::Mat hiddenMask = occluderMask(occluders, image.size());
    std::cout << "OCCLUDERS: " << occluders.size() << std::endl;
    std::vector<CircleModel> rescored;
    for (CircleModel m : proposals)
    {
        scoreCircle(m, nearBorder, gradX, gradY, ballColor, hiddenMask, p);
        if (m.visible < p.minVisible || m.edgeSupport < p.minEdgeSupport || m.insideColor < p.minInsideColor)
            continue;

        // Occluded circle must have typical size
        bool occluded = m.visible < p.occludedVisible;
        if (occluded && (m.radius < p.occludedMinRatio * medianRadius || m.radius > p.occludedMaxRatio * medianRadius))
            continue;
        rescored.push_back(m);
    }
    std::vector<CircleModel> chosen = selectCircles(rescored, p, radiusLow, radiusHigh);
    std::cout << "CHOSEN: " << chosen.size() << std::endl;

    //9.Final check
    std::vector<DroppedCircle> dropped;
    std::vector<CircleModel> balls = verifyCircles(chosen, image, ballColor, p, dropped);

    //10.Draw circles and boxes
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

    //11.Save and show
    saveProcessImages(image, greenness, gradX, gradY, ballColor, borderMap, groups, leftover, occluders, chosen, dropped, result, show);
    std::cout << "Completed" << std::endl;
    return 0;
}
