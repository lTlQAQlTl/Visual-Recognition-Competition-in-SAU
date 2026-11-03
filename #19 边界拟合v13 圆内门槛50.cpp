#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>

//0.判断标准
struct Params
{
    //白平衡
    int whiteSaturationMax = 85;
    int whiteBrightnessMin = 60;
    int whiteBrightnessMax = 248;
    double whiteBrightnessShare = 0.70;
    int whiteSampleStep = 12;
    int whitePatchRadius = 2;
    int whiteMinSamples = 12;
    double whiteBinWidth = 0.02;
    double whiteMinGain = 0.60;
    double whiteMaxGain = 1.70;

    //颜色
    int hueMin = 33;
    int hueMax = 56;
    int saturationMin = 15;
    int saturationMax = 125;
    double brightnessMinShare = 0.30;

    //明度修正
    int valueTarget = 200;
    double valuePercentile = 0.85;
    double gammaMin = 0.75;
    double gammaMax = 1.25;

    //亮度
    double darkShare = 0.20;
    double fullShare = 0.60;

    //动态
    int brightnessMin = 45;
    int darkLevel = 30;
    int fullLevel = 90;

    //边界
    double gradientMin = 0.0035;
    int sideOffset = 3;

    //拟合圆
    double fitError = 3.0;
    float radiusMin = 6.0f;
    double minRadiusFraction = 1.0 / 70;
    int maxIterations = 4000;
    int refineIterations = 30000;
    double refineWindow = 2.2;
    double refineBridge = 0.6;
    double firstInlierShare = 0.25;
    double refineInlierShare = 0.15;

    //阴影补判
    double shadowSeedShare = 0.30;
    double shadowMinDark = 0.15;
    double shadowMaxDark = 0.75;
    int shadowHueDifference = 8;
    int shadowSaturationDifference = 35;
    double shadowVote = 0.65;

    //给圆评分
    int perimeterSamples = 72;
    double edgeSearch = 3.0;
    double minEdgeSupport = 0.30;
    double minInsideColor = 0.50;

    //选择
    double overlapRatio = 0.75;
    double referenceEdgeSupport = 0.60;
    double minRadiusRatio = 0.85;
    double maxRadiusRatio = 1.20;

    //检查
    double finalInsideColor = 0.50;
    double maxUnexplainedOutside = 0.30;
    double trustedEdgeSupport = 0.50;

    //反向闭合检测
    double minReverseSupport = 0.25;
    double maxOccluderInside = 0.20;
    double minOccluderRatio = 0.70;
    double maxOccluderRatio = 1.30;
    double minVisible = 0.25;
    double occludedVisible = 0.90;
    double occludedMinRatio = 0.95;
    double occludedMaxRatio = 1.05;
    double maxUnexplainedTrusted = 0.50;
    double maxUnexplainedOccluded = 0.20;
};
//保存候选圆
struct CircleModel
{
    cv::Point2f center;
    float radius = 0.0f;
    double edgeSupport = 0.0;
    double insideColor = 0.0;
    double shadowColor = 0.0;
    double reverseSupport = 0.0;
    double visible = 1.0;
};

//1.白色采样去偏色
cv::Mat whiteBalance(
    const cv::Mat& image,
    const Params& p,
    double& medianBrightness,
    cv::Mat& sampleView)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv,
        cv::COLOR_BGR2HSV);
    auto percentile = [](const cv::Mat& m, double share)
    {
        int hist[256] = {};
        for (int y = 0; y < m.rows; ++y)
            for (int x = 0; x < m.cols; ++x) ++hist[m.at<uchar>(y, x)];
        int target = int((m.total() - 1) * share), sum = 0;
        for (int v = 0; v < 256; ++v)
            if ((sum += hist[v]) > target) return double(v);
        return 255.0;
    };
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    int brightnessMin = std::max(p.whiteBrightnessMin,
        int(p.whiteBrightnessShare * percentile(channels[2], 0.85)));

    //亮而未过曝，色相不限
    cv::Mat candidate;
    cv::inRange(hsv,
        cv::Scalar(0, 0, brightnessMin),
        cv::Scalar(179, p.whiteSaturationMax, p.whiteBrightnessMax),
        candidate);

    //转线性RGB，避免直接在编码值上估算光照增益
    float linear[256];
    for (int v = 0; v < 256; ++v)
    {
        float s = v / 255.0f;
        linear[v] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    }

    //每个小块只投一票，减少噪点和边缘混色
    std::vector<cv::Point> points;
    std::vector<cv::Vec3f> colors;
    int bins = int(std::ceil(1.0 / p.whiteBinWidth)) + 1;
    cv::Mat hist = cv::Mat::zeros(bins, bins, CV_32S);
    int radius = p.whitePatchRadius;
    int area = (2 * radius + 1) * (2 * radius + 1);
    for (int y = radius; y < image.rows - radius; y += p.whiteSampleStep)
    {
        for (int x = radius; x < image.cols - radius; x += p.whiteSampleStep)
        {
            if (!candidate.at<uchar>(y, x)) continue;
            cv::Vec3f color(0, 0, 0);
            int count = 0;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    if (!candidate.at<uchar>(y + dy, x + dx)) continue;
                    cv::Vec3b c = image.at<cv::Vec3b>(y + dy, x + dx);
                    cv::Vec3f f(linear[c[0]], linear[c[1]], linear[c[2]]);
                    color += f / (f[0] + f[1] + f[2]);
                    ++count;
                }
            }
            if (count < 0.8 * area) continue;
            color /= float(count);
            points.push_back(cv::Point(x, y));
            colors.push_back(color);
            int b = std::clamp(int(color[0] / p.whiteBinWidth), 0, bins - 1);
            int r = std::clamp(int(color[2] / p.whiteBinWidth), 0, bins - 1);
            ++hist.at<int>(b, r);
        }
    }

    //选通道比例最集中的一组，剔除不同颜色的候选块
    int best = 0, bestB = 0, bestR = 0;
    for (int b = 1; b < bins - 1; ++b)
    {
        for (int r = 1; r < bins - 1; ++r)
        {
            int votes = 0;
            for (int db = -1; db <= 1; ++db)
                for (int dr = -1; dr <= 1; ++dr) votes += hist.at<int>(b + db, r + dr);
            if (votes > best) { best = votes; bestB = b; bestR = r; }
        }
    }
    sampleView = image.clone();
    std::vector<float> values[3];
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        int b = int(colors[i][0] / p.whiteBinWidth);
        int r = int(colors[i][2] / p.whiteBinWidth);
        bool selected = std::abs(b - bestB) <= 1 && std::abs(r - bestR) <= 1;
        cv::circle(sampleView, points[i], selected ? 3 : 1,
            selected ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255),
            cv::FILLED);
        if (selected)
            for (int c = 0; c < 3; ++c) values[c].push_back(colors[i][c]);
    }

    double gain[3] = { 1.0, 1.0, 1.0 };
    bool valid = values[0].size() >= std::size_t(p.whiteMinSamples);
    if (valid)
    {
        double white[3];
        for (int c = 0; c < 3; ++c)
        {
            auto& v = values[c];
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            white[c] = v[v.size() / 2];
        }
        //保持参照白的线性亮度，只校正通道比例
        double target = 0.0722 * white[0] + 0.7152 * white[1] + 0.2126 * white[2];
        for (int c = 0; c < 3; ++c)
        {
            gain[c] = target / std::max(white[c], 1e-6);
            if (gain[c] < p.whiteMinGain || gain[c] > p.whiteMaxGain) valid = false;
        }
    }
    if (!valid)
        for (double& g : gain) g = 1.0;

    //查表校正，再转回sRGB
    uchar table[3][256];
    for (int c = 0; c < 3; ++c)
    {
        for (int v = 0; v < 256; ++v)
        {
            double f = linear[v] * gain[c];
            double s = f <= 0.0031308 ? 12.92 * f : 1.055 * std::pow(f, 1.0 / 2.4) - 0.055;
            table[c][v] = cv::saturate_cast<uchar>(255.0 * s);
        }
    }
    cv::Mat balanced(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            balanced.at<cv::Vec3b>(y, x) = cv::Vec3b(table[0][c[0]], table[1][c[1]], table[2][c[2]]);
        }
    cv::Mat brightness;
    cv::split(balanced, channels);
    cv::max(channels[0], channels[1], brightness);
    cv::max(brightness, channels[2], brightness);
    medianBrightness = percentile(brightness, 0.50);
    std::cout << "WHITE CANDIDATES: " << points.size() << std::endl;
    std::cout << "WHITE SAMPLES: " << values[0].size() << std::endl;
    std::cout << "WHITE GAIN BGR: " << gain[0] << " " << gain[1] << " " << gain[2] << std::endl;
    std::cout << "WHITE STATUS: " << (valid ? "applied" : "skipped") << std::endl;
    return balanced;
}

//明度修正
cv::Mat correctValue(
    const cv::Mat& image,
    const Params& p,
    double& medianBrightness)
{
    int hist[256] = {};
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            ++hist[std::max({ c[0], c[1], c[2] })];
        }
    auto percentile = [&](double share)
    {
        auto target = std::size_t((image.total() - 1) * share);
        std::size_t sum = 0;
        for (int v = 0; v < 256; ++v)
            if ((sum += hist[v]) > target) return v;
        return 255;
    };

    //用较亮区域确定曲线，限制极暗和接近过曝时的调整
    int reference = percentile(p.valuePercentile);
    double gamma = 1.0;
    if (reference >= 32 && reference <= 248)
        gamma = std::clamp(std::log(p.valueTarget / 255.0) / std::log(reference / 255.0),
            p.gammaMin, p.gammaMax);

    //按同一个比例调整三个通道，保留HSV色相和饱和度
    double scale[256] = {};
    for (int v = 1; v < 256; ++v)
        scale[v] = std::pow(v / 255.0, gamma - 1.0);
    cv::Mat corrected(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3b c = image.at<cv::Vec3b>(y, x);
            double gain = scale[std::max({ c[0], c[1], c[2] })];
            corrected.at<cv::Vec3b>(y, x) = cv::Vec3b(
                cv::saturate_cast<uchar>(c[0] * gain),
                cv::saturate_cast<uchar>(c[1] * gain),
                cv::saturate_cast<uchar>(c[2] * gain));
        }
    int median = percentile(0.50);
    medianBrightness = cvRound(median * scale[median]);
    std::cout << "VALUE REFERENCE: " << reference << std::endl;
    std::cout << "VALUE GAMMA: " << gamma << std::endl;
    return corrected;
}

//2.绿色程度图
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

    //降噪
    cv::GaussianBlur(greenness, greenness,
        cv::Size(9, 9),
        2.0);
    return greenness;
}

//3.检查绿球颜色
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
            cv::Vec3b v = hsv.at<cv::Vec3b>(y, x);
            int hue = v[0], saturation = v[1];
            int brightness = v[2];
            if (hue >= p.hueMin && hue <= p.hueMax && saturation >= p.saturationMin && saturation <= p.saturationMax && brightness >= p.brightnessMin)
                ballColor.at<uchar>(y, x) = 255;
        }
    }
    return ballColor;
}

//4.寻找拟合点
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

            //保留变化峰值
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

            //保证绿色一侧为绿球
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

//5.分组
std::vector<std::vector<cv::Point2f>> groupBorderPoints(
    const cv::Mat& borderMap,
    int minPoints,
    int bridge = 5)
{
    //连线
    cv::Mat bridged, labels;
    cv::dilate(borderMap, bridged,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(bridge, bridge)));
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

//6.三点寻圆
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

//7.不断找点修正拟合
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

//8.打分
//候选圆内的阴影补判
double shadowVotes(
    const CircleModel& c,
    const cv::Mat& ballColor,
    const cv::Mat& hsv,
    const cv::Mat& luminance,
    const cv::Mat& occluders,
    const std::vector<cv::Point>& samples,
    const Params& p)
{
    constexpr int half = 10;
    constexpr int size = 2 * half + 1;
    double step = c.radius / half;
    cv::Mat seed = cv::Mat::zeros(size, size, CV_8U);
    cv::Mat visible = seed.clone();
    auto pointAt = [&](int x, int y)
    {
        return cv::Point(cvRound(c.center.x + (x - half) * step),
            cvRound(c.center.y + (y - half) * step));
    };
    int area = 0, green = 0;
    for (int y = 1; y < size - 1; ++y)
        for (int x = 1; x < size - 1; ++x)
        {
            int dx = x - half, dy = y - half;
            if (dx * dx + dy * dy > 81) continue;
            cv::Point q = pointAt(x, y);
            if (q.x < 0 || q.x >= hsv.cols || q.y < 0 || q.y >= hsv.rows) continue;
            if (!occluders.empty() && occluders.at<uchar>(q)) continue;
            visible.at<uchar>(y, x) = 255;
            seed.at<uchar>(y, x) = ballColor.at<uchar>(q);
            ++area;
            if (seed.at<uchar>(y, x)) ++green;
        }
    if (!area || green <= p.shadowSeedShare * area) return 0.0;

    //只用最大连续绿区作参照
    cv::Mat labels, stats, centers;
    int count = cv::connectedComponentsWithStats(seed, labels, stats, centers, 8);
    int best = 1;
    for (int i = 2; i < count; ++i)
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(best, cv::CC_STAT_AREA)) best = i;
    if (count < 2 || stats.at<int>(best, cv::CC_STAT_AREA) <= p.shadowSeedShare * area) return 0.0;
    cv::Mat connectedSeed = labels == best;
    std::vector<double> hs, ss, ys;
    for (int y = 1; y < size - 1; ++y)
        for (int x = 1; x < size - 1; ++x)
        {
            if (!connectedSeed.at<uchar>(y, x)) continue;
            cv::Point q = pointAt(x, y);
            cv::Vec3b v = hsv.at<cv::Vec3b>(q);
            hs.push_back(v[0]);
            ss.push_back(v[1]);
            ys.push_back(luminance.at<float>(q));
        }
    auto median = [](std::vector<double>& v)
    {
        auto mid = v.begin() + v.size() / 2;
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    };
    double h = median(hs), s = median(ss), light = median(ys);
    if (light <= 0.0) return 0.0;

    //用相对亮度估算阴影，只对颜色相近的暗点作有限修正
    auto shadow = [&](cv::Point q)
    {
        double ratio = luminance.at<float>(q) / light;
        if (ratio < 1.0 - p.shadowMaxDark || ratio > 1.0 - p.shadowMinDark) return false;
        cv::Vec3b v = hsv.at<cv::Vec3b>(q);
        if (std::abs(v[0] - h) > p.shadowHueDifference || std::abs(v[1] - s) > p.shadowSaturationDifference) return false;
        double mix = std::min(0.5, 1.0 - ratio);
        double hh = v[0] + mix * (h - v[0]), correctedS = v[1] + mix * (s - v[1]);
        if (hh < p.hueMin || hh > p.hueMax || correctedS < p.saturationMin || correctedS > p.saturationMax)
            return false;
        if (v[2] >= p.brightnessMin) return true;
        auto linearValue = [](int value)
        {
            double v = value / 255.0;
            return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
        };
        return linearValue(v[2]) / ratio >= linearValue(p.brightnessMin);
    };
    cv::Mat region = connectedSeed.clone();
    for (int y = 1; y < size - 1; ++y)
        for (int x = 1; x < size - 1; ++x)
            if (visible.at<uchar>(y, x) && shadow(pointAt(x, y))) region.at<uchar>(y, x) = 255;
    cv::connectedComponents(region, labels, 8);
    std::vector<cv::Point> roots;
    cv::findNonZero(connectedSeed, roots);
    int root = labels.at<int>(roots.front());
    int votes = 0;
    for (const auto& q : samples)
    {
        int x = cvRound((q.x - c.center.x) / step) + half;
        int y = cvRound((q.y - c.center.y) / step) + half;
        if (x < 0 || y < 0 || x >= size || y >= size) continue;
        if (labels.at<int>(y, x) == root && shadow(q)) ++votes;
    }
    return votes * p.shadowVote;
}

void scoreCircle(
    CircleModel& c,
    const cv::Mat& nearBorder,
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& ballColor,
    const cv::Mat& hsv,
    const cv::Mat& luminance,
    const cv::Mat& occluders,
    const Params& p)
{
    int rows = ballColor.rows, cols = ballColor.cols;
    // Hidden by an occluder or the picture edge
    auto hidden = [&](int x, int y)
    {
        if (x < 0 || y < 0 || x >= cols || y >= rows) return true;
        return !occluders.empty() && occluders.at<uchar>(y, x) != 0;
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
    std::vector<cv::Point> failed;
    //圆心一票，内圈等距分布，采样数随周长增加
    for (int ring = 0; ring <= 4; ++ring)
    {
        double rr = 0.175 * ring;
        int n = ring == 0 ? 1 : 8 * ring;
        for (int k = 0; k < n; ++k)
        {
            double a = 2.0 * CV_PI * k / n;
            int x = cvRound(c.center.x + rr * c.radius * std::cos(a));
            int y = cvRound(c.center.y + rr * c.radius * std::sin(a));
            if (hidden(x, y)) continue;
            ++checked;
            if (ballColor.at<uchar>(y, x)) ++inside;
            else failed.emplace_back(x, y);
        }
    }
    c.shadowColor = 0.0;
    c.insideColor = checked ? double(inside) / checked : 0.0;
    if (checked && c.edgeSupport >= p.minEdgeSupport && c.insideColor < p.finalInsideColor)
    {
        c.shadowColor = shadowVotes(c, ballColor, hsv, luminance, occluders, failed, p) / checked;
        c.insideColor += c.shadowColor;
    }
}

//9.RANSAC生成大量候选圆
std::vector<CircleModel> proposeCircles(
    const std::vector<cv::Point2f>& pts,
    float radiusLow,
    float radiusHigh,
    double window,
    int iterationsPerPoint,
    int maxIterations,
    double minInlierShare,
    const cv::Mat& nearBorder,
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& ballColor,
    const cv::Mat& hsv,
    const cv::Mat& luminance,
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

        //找绿侧点
        std::vector<cv::Point2f> inward, outward;
        for (const auto& q : pts)
        {
            double d = cv::norm(q - m.center);
            if (std::abs(d - m.radius) > p.fitError || d < 1e-6) continue;
            float gx = gradX.at<float>(cvRound(q.y), cvRound(q.x));
            float gy = gradY.at<float>(cvRound(q.y), cvRound(q.x));
            float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < 1e-6f) continue;
            double cosine = -(gx * (q.x - m.center.x) + gy * (q.y - m.center.y)) / (d * mag);
            if (cosine > 0.5) inward.push_back(q);
            else if (cosine < -0.5) outward.push_back(q);
        }
        std::vector<cv::Point2f>& inliers = (inward.size() >= outward.size()) ? inward : outward;

        //判断符合点数
        if (inliers.size() < std::max<std::size_t>(8, std::size_t(minInlierShare * 2 * CV_PI * m.radius)))
            continue;
        if (!fitCircleLeastSquares(inliers, m.center, m.radius)) continue;
        if (m.radius < radiusLow || m.radius > radiusHigh) continue;
        scoreCircle(m, nearBorder, gradX, gradY, ballColor, hsv, luminance,
            cv::Mat(),
            p);
        if (m.edgeSupport >= p.minEdgeSupport && m.insideColor >= p.minInsideColor)
            proposals.push_back(m);
        else if (reverseCircles && m.reverseSupport >= p.minReverseSupport && m.insideColor <= p.maxOccluderInside)
            reverseCircles->push_back(m);
    }
    return proposals;
}

//10.候选圆最终检查
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
    const cv::Mat& ballColor,
    const Params& p,
    std::vector<DroppedCircle>& dropped)
{
    std::vector<CircleModel> kept;
    for (const auto& c : circles)
    {
        if (c.insideColor < p.finalInsideColor)
            dropped.push_back({ c, "inside" });
        else
            kept.push_back(c);
    }

    //第一次筛选
    while (true)
    {
        double worst = 0.0;
        int worstIndex = -1;
        for (std::size_t i = 0; i < kept.size(); ++i)
        {
            // Occluded circle uses a strict limit
            double limit;
            if (kept[i].visible < p.occludedVisible) limit = p.maxUnexplainedOccluded;
            else if (kept[i].edgeSupport >= p.trustedEdgeSupport) limit = p.maxUnexplainedTrusted;
            else limit = p.maxUnexplainedOutside;
            double u = unexplainedOutside(i, kept, ballColor);
            if (u > limit && u - limit > worst) { worst = u - limit; worstIndex = int(i); }
        }
        if (worstIndex < 0) break;
        dropped.push_back({ kept[worstIndex], "outside" });
        kept.erase(kept.begin() + worstIndex);
    }
    return kept;
}

//11.选圆
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

//12.选出不冲突圆
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

//遮挡测试
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

//13.保存过程图片
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
    const std::string& prefix,
    bool show)
{
    std::vector<std::pair<std::string, cv::Mat>> steps;

    //0.明度修正
    steps.push_back({ "0_value_correction", image });

    //1.绿色热力图
    cv::Mat green8;
    greenness.convertTo(green8, CV_8U, 1500.0, 60.0);
    steps.push_back({ "1_greenness", green8 });

    //2.球颜色位置
    cv::Mat colorView = dimmed(image);
    colorView.setTo(cv::Scalar(255,0,255),
        ballColor);
    steps.push_back({ "2_ball_color", colorView });

    //3.绿色变化率
    cv::Mat magnitude, rate8;
    cv::magnitude(gradX, gradY, magnitude);
    magnitude.convertTo(rate8, CV_8U, 20000.0);
    steps.push_back({ "3_change_rate", rate8 });

    //4.保留边界点
    cv::Mat borderView = dimmed(image);
    borderView.setTo(cv::Scalar(0,0,255),
        borderMap);
    steps.push_back({ "4_border_points", borderView });

    //5.边界分组
    cv::Mat groupView = dimmed(image);
    cv::RNG colorRng(7);
    for (const auto& g : groups)
    {
        cv::Vec3b color(uchar(colorRng.uniform(80, 256)), uchar(colorRng.uniform(80, 256)), uchar(colorRng.uniform(80, 256)));
        for (const auto& q : g)
            groupView.at<cv::Vec3b>(cvRound(q.y), cvRound(q.x)) = color;
    }
    steps.push_back({ "5_border_groups", groupView });

    //6.遮挡关系
    cv::Mat occluderView = dimmed(image);
    occluderView.setTo(cv::Scalar(0,0,255),
        leftover);
    for (const auto& o : occluders) drawCircle(occluderView, o, cv::Scalar(0,255,255), 2);
    steps.push_back({ "6_occluders", occluderView });

    //7.候选
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

    //8.结果
    steps.push_back({ "8_result", result });
    for (const auto& s : steps)
    {
        cv::imwrite(prefix + s.first + ".png", s.second);
        if (show) cv::imshow(prefix + s.first, s.second);
    }
}

//从第一轮可信圆的内部重新取色
bool resampleColor(
    const cv::Mat& hsv,
    const std::vector<CircleModel>& balls,
    Params& p)
{
    std::vector<int> hueLow, hueHigh, saturationHigh;
    for (const auto& c : balls)
    {
        if (c.edgeSupport < 0.50 || c.insideColor - c.shadowColor < 0.75 || c.visible < 0.90)
            continue;
        std::vector<int> hues, saturations;
        for (int dy = -5; dy <= 5; ++dy)
            for (int dx = -5; dx <= 5; ++dx)
            {
                if (dx * dx + dy * dy > 25) continue;
                int x = cvRound(c.center.x + 0.5 * c.radius * dx / 5.0);
                int y = cvRound(c.center.y + 0.5 * c.radius * dy / 5.0);
                if (x < 0 || x >= hsv.cols || y < 0 || y >= hsv.rows) continue;
                auto v = hsv.at<cv::Vec3b>(y, x);
                if (v[2] < p.brightnessMin || v[2] >= 248 || v[1] < p.saturationMin ||
                    v[0] < p.hueMin - 8 || v[0] > p.hueMax + 8) continue;
                hues.push_back(v[0]);
                saturations.push_back(v[1]);
            }
        if (hues.size() < 40) continue;
        std::sort(hues.begin(), hues.end());
        std::sort(saturations.begin(), saturations.end());
        hueLow.push_back(hues[(hues.size() - 1) / 20]);
        hueHigh.push_back(hues[(hues.size() - 1) * 19 / 20]);
        saturationHigh.push_back(saturations[(saturations.size() - 1) * 19 / 20]);
    }
    std::cout << "RESAMPLE BALLS: " << hueLow.size() << std::endl;
    if (hueLow.size() < 3) return false;
    auto median = [](std::vector<int>& v)
    {
        auto mid = v.begin() + v.size() / 2;
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    };
    int low = std::clamp(median(hueLow) - 2, std::max(0, p.hueMin - 8), p.hueMin);
    int high = std::clamp(median(hueHigh) + 2, p.hueMax, std::min(179, p.hueMax + 8));
    int sat = std::clamp(median(saturationHigh) + 5, p.saturationMax, std::min(255, p.saturationMax + 35));
    std::cout << "RESAMPLE RANGE: H " << low << " " << high
        << " S " << p.saturationMin << " " << sat << std::endl;
    bool changed = low != p.hueMin || high != p.hueMax || sat != p.saturationMax;
    p.hueMin = low;
    p.hueMax = high;
    p.saturationMax = sat;
    return changed;
}

cv::Mat drawResult(const cv::Mat& original, const std::vector<CircleModel>& balls)
{
    cv::Mat result = original.clone();
    cv::Rect imageArea(0, 0, original.cols, original.rows);
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
    return result;
}

//重复使用同一套边界、遮挡和拟合流程
std::vector<CircleModel> detectBalls(
    const cv::Mat& image,
    const cv::Mat& original,
    const cv::Mat& greenness,
    const cv::Mat& gradX,
    const cv::Mat& gradY,
    const cv::Mat& hsv,
    const cv::Mat& luminance,
    Params p,
    const std::string& prefix,
    bool show)
{
    cv::Mat ballColor = computeBallColor(image, p);
    //4.边界图
    cv::Mat borderMap;
    std::vector<cv::Point2f> border = findBorderPoints(gradX, gradY, ballColor, p, borderMap);
    cv::Mat nearBorder;
    int k = 2 * int(p.edgeSearch) + 1;
    cv::dilate(borderMap, nearBorder,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    std::cout << "BORDER POINTS: " << border.size() << std::endl;

    //5.分组
    std::vector<std::vector<cv::Point2f>> groups = groupBorderPoints(borderMap, 20);
    std::cout << "BORDER GROUPS: " << groups.size() << std::endl;

    //6.首次拟合
    float radiusMax = std::min(image.cols, image.rows) / 8.0f;
    p.radiusMin = std::max(6.0f, float(std::min(image.cols, image.rows) * p.minRadiusFraction));
    std::mt19937 rng(42);
    std::vector<CircleModel> proposals;
    for (const auto& g : groups)
    {
        auto part = proposeCircles(g, p.radiusMin, radiusMax, 2.0 * radiusMax, 4, p.maxIterations, p.firstInlierShare, nearBorder, gradX, gradY, ballColor, hsv, luminance, p, rng);
        proposals.insert(proposals.end(), part.begin(), part.end());
    }
    std::cout << "PROPOSALS: " << proposals.size() << std::endl;

    //7.典型半径
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
    //8.二次拟合
    float radiusLow = float(p.minRadiusRatio * medianRadius);
    float radiusHigh = float(p.maxRadiusRatio * medianRadius);
    float occluderLow = float(p.minOccluderRatio * medianRadius);
    float occluderHigh = float(p.maxOccluderRatio * medianRadius);
    std::vector<CircleModel> reverse;
    //隐藏球边界
    int bridge = std::max(5, int(p.refineBridge * medianRadius) | 1);
    std::vector<std::vector<cv::Point2f>> refineGroups = groupBorderPoints(borderMap, 20, bridge);
    for (const auto& g : refineGroups)
    {
        auto part = proposeCircles(g,
            std::min(radiusLow, occluderLow),
            std::max(radiusHigh, occluderHigh),
            p.refineWindow * medianRadius, 8, p.refineIterations, p.refineInlierShare, nearBorder, gradX, gradY, ballColor, hsv, luminance, p, rng, &reverse);
        proposals.insert(proposals.end(), part.begin(), part.end());
    }

    //9.遮挡关系
    std::vector<CircleModel> plainChosen = selectCircles(proposals, p, radiusLow, radiusHigh);
    cv::Mat leftover = leftoverBorder(borderMap, plainChosen, p);
    cv::Mat nearLeftover;
    cv::dilate(leftover, nearLeftover,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    std::vector<CircleModel> leftoverReverse;
    for (CircleModel m : reverse)
    {
        scoreCircle(m, nearLeftover, gradX, gradY, ballColor, hsv, luminance,
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
        scoreCircle(m, nearBorder, gradX, gradY, ballColor, hsv, luminance, hiddenMask, p);
        if (m.visible < p.minVisible || m.edgeSupport < p.minEdgeSupport || m.insideColor < p.minInsideColor)
            continue;

        //圆心位置取舍
        if (m.center.x < 0 || m.center.y < 0 || m.center.x >= image.cols || m.center.y >= image.rows)
            continue;

        //球大小
        bool occluded = m.visible < p.occludedVisible;
        if (occluded && (m.radius < p.occludedMinRatio * medianRadius || m.radius > p.occludedMaxRatio * medianRadius))
            continue;
        rescored.push_back(m);
    }
    std::vector<CircleModel> chosen = selectCircles(rescored, p, radiusLow, radiusHigh);
    std::cout << "CHOSEN: " << chosen.size() << std::endl;

    //10.检查
    std::vector<DroppedCircle> dropped;
    std::vector<CircleModel> balls = verifyCircles(chosen, ballColor, p, dropped);

    std::cout << "SHADOW HELPED: " << std::count_if(balls.begin(), balls.end(), [&](const CircleModel& c)
    {
        return c.shadowColor > 0 && c.insideColor - c.shadowColor < p.finalInsideColor;
    }) << std::endl;

    std::cout << "MEDIAN RADIUS: " << medianRadius << std::endl;
    cv::Mat result = drawResult(original, balls);
    saveProcessImages(image, greenness, gradX, gradY, ballColor, borderMap, groups,
        leftover, occluders, chosen, dropped, result, prefix, show);
    if (prefix.empty()) cv::imwrite("8_first_result.png", result);
    return balls;
}

int main(int argc, char** argv)
{
    //1.读图
    std::string path = (argc > 1) ? argv[1] : "test_output.png";
    bool show = (argc > 2) && std::string(argv[2]) == "--show";
    cv::Mat image = cv::imread(path);
    if (image.empty())
    {
        std::cout << "Failed to read " << path << std::endl;
        return 1;
    }
    Params p;

    //2.白平衡
    cv::Mat original = image;
    double medianBrightness = 0.0;
    cv::Mat whiteSamples;
    image = whiteBalance(original, p, medianBrightness, whiteSamples);
    cv::Mat balanced = image;
    image = correctValue(balanced, p, medianBrightness);
    p.brightnessMin = int(p.brightnessMinShare * medianBrightness);
    p.darkLevel = int(p.darkShare * medianBrightness);
    p.fullLevel = std::max(p.darkLevel + 1, int(p.fullShare * medianBrightness));

    //3.绿色程度
    cv::Mat greenness = computeGreenness(image, p);
    cv::Mat gradX, gradY;
    cv::Sobel(greenness, gradX, CV_32F, 1, 0, 3, 1.0 / 8);
    cv::Sobel(greenness, gradY, CV_32F, 0, 1, 3, 1.0 / 8);
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    float linear[256];
    for (int i = 0; i < 256; ++i)
    {
        float v = i / 255.0f;
        linear[i] = v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    }
    cv::Mat luminance(image.size(), CV_32F);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
        {
            auto c = image.at<cv::Vec3b>(y, x);
            luminance.at<float>(y, x) = 0.0722f * linear[c[0]] + 0.7152f * linear[c[1]] + 0.2126f * linear[c[2]];
        }

    //4.第一次识别
    std::vector<CircleModel> balls = detectBalls(image, original, greenness, gradX, gradY, hsv, luminance, p, "", show);
    std::cout << "FIRST PASS: " << balls.size() << std::endl;

    //5.再次取色，只补充第一次漏掉的球
    std::size_t firstCount = balls.size();
    Params second = p;
    if (resampleColor(hsv, balls, second))
    {
        auto extra = detectBalls(image, original, greenness, gradX, gradY, hsv, luminance, second, "resample_", show);
        std::vector<float> radii;
        for (const auto& c : balls) radii.push_back(c.radius);
        auto mid = radii.begin() + radii.size() / 2;
        std::nth_element(radii.begin(), mid, radii.end());
        float radius = *mid;
        for (const auto& c : extra)
        {
            if (c.radius < p.minRadiusRatio * radius || c.radius > p.maxRadiusRatio * radius) continue;
            bool overlap = std::any_of(balls.begin(), balls.end(), [&](const CircleModel& old)
            {
                return cv::norm(c.center - old.center) < p.overlapRatio * (c.radius + old.radius);
            });
            if (!overlap) balls.push_back(c);
        }
    }
    std::cout << "RESAMPLE ADDED: " << balls.size() - firstCount << std::endl;

    //11.画圆
    cv::Mat result = drawResult(original, balls);
    std::cout << "GREEN BALLS: " << balls.size() << std::endl;

    //12.保存
    cv::imwrite("0_white_samples.png", whiteSamples);
    cv::imwrite("0_white_balance.png", balanced);
    if (show) cv::imshow("0_white_balance", balanced);
    if (show) cv::imshow("0_white_samples", whiteSamples);
    cv::imwrite("8_result.png", result);
    if (show)
    {
        cv::imshow("8_result", result);
        cv::waitKey(0);
    }
    std::cout << "Completed" << std::endl;
    return 0;
}
