// OpenCV
#include <opencv2/opencv.hpp>

// C++ STL
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

namespace fs = std::filesystem;

namespace {

cv::Mat buildLaneColorMask(const cv::Mat& bgrImage) {
    cv::Mat hls;
    cv::cvtColor(bgrImage, hls, cv::COLOR_BGR2HLS);

    // White mask in HLS
    cv::Mat whiteMask;
    cv::inRange(hls, cv::Scalar(0, 150, 0), cv::Scalar(180, 255, 255), whiteMask);

    // Yellow mask in HLS (tuned ranges)
    cv::Mat yellowMask;
    cv::inRange(hls, cv::Scalar(8, 70, 70), cv::Scalar(50, 255, 255), yellowMask);

    cv::Mat laneMask;
    cv::bitwise_or(whiteMask, yellowMask, laneMask);
    return laneMask;
}

cv::Mat applyRegionOfInterestMask(const cv::Mat& edgeImage) {
    const int width = edgeImage.cols;
    const int height = edgeImage.rows;

    std::vector<cv::Point> vertices;
    vertices.emplace_back(static_cast<int>(0.15 * width), height);
    vertices.emplace_back(static_cast<int>(0.45 * width), static_cast<int>(0.62 * height));
    vertices.emplace_back(static_cast<int>(0.55 * width), static_cast<int>(0.62 * height));
    vertices.emplace_back(static_cast<int>(0.85 * width), height);

    cv::Mat mask = cv::Mat::zeros(edgeImage.size(), edgeImage.type());
    std::vector<std::vector<cv::Point>> pts{vertices};
    cv::fillPoly(mask, pts, cv::Scalar(255));

    cv::Mat masked;
    cv::bitwise_and(edgeImage, mask, masked);
    return masked;
}

struct LineModel {
    double slope{0.0};
    double intercept{0.0};
};

struct ClassifiedLines {
    std::vector<cv::Vec4i> left;
    std::vector<cv::Vec4i> right;
};

ClassifiedLines classifyAndFilterLines(const std::vector<cv::Vec4i>& lines, int imageWidth, int imageHeight, int yRef) {
    ClassifiedLines result;
    const double midX = static_cast<double>(imageWidth) * 0.5;

    for (const cv::Vec4i& l : lines) {
        const double x1 = l[0];
        const double y1 = l[1];
        const double x2 = l[2];
        const double y2 = l[3];
        const double dx = x2 - x1;
        const double dy = y2 - y1;
        if (std::abs(dx) < 1e-6) {
            continue;
        }
        const double slope = dy / dx;
        if (std::abs(slope) < 0.2) {
            continue; // near-horizontal
        }
        const double intercept = y1 - slope * x1;
        const double xAtRef = (yRef - intercept) / slope;
        if (xAtRef < 0.05 * imageWidth || xAtRef > 0.95 * imageWidth) {
            continue; // unreasonable bottom intersection
        }

        if (slope < 0.0 && xAtRef < midX) {
            result.left.push_back(l);
        } else if (slope > 0.0 && xAtRef > midX) {
            result.right.push_back(l);
        }
    }
    return result;
}

std::optional<LineModel> fitAverageLine(const std::vector<cv::Vec4i>& lines, bool wantNegativeSlope) {
    double slopeSum = 0.0;
    double interceptSum = 0.0;
    double weightSum = 0.0;

    for (const cv::Vec4i& l : lines) {
        const double x1 = l[0];
        const double y1 = l[1];
        const double x2 = l[2];
        const double y2 = l[3];

        const double dx = x2 - x1;
        const double dy = y2 - y1;
        if (std::abs(dx) < 1e-6) {
            continue;
        }
        const double slope = dy / dx;
        // Filter out nearly horizontal lines (relaxed to catch dashed lanes)
        if (std::abs(slope) < 0.3) {
            continue;
        }
        if (wantNegativeSlope && slope >= 0.0) {
            continue;
        }
        if (!wantNegativeSlope && slope <= 0.0) {
            continue;
        }
        const double intercept = y1 - slope * x1;
        const double length = std::hypot(dx, dy);
        slopeSum += slope * length;
        interceptSum += intercept * length;
        weightSum += length;
    }

    if (weightSum <= 0.0) {
        return std::nullopt;
    }
    LineModel model;
    model.slope = slopeSum / weightSum;
    model.intercept = interceptSum / weightSum;
    return model;
}

void drawLineForYRange(cv::Mat& image, const LineModel& model, int yMin, int yMax, const cv::Scalar& color, int thickness) {
    // x = (y - b) / m
    auto computeX = [&](int y) -> int {
        return static_cast<int>((y - model.intercept) / model.slope);
    };
    const cv::Point p1(computeX(yMax), yMax);
    const cv::Point p2(computeX(yMin), yMin);
    cv::line(image, p1, p2, color, thickness, cv::LINE_AA);
}

// Draws a slightly extended line beyond the ROI upwards to improve visual continuity
void drawExtendedLine(cv::Mat& image, const LineModel& model, int yMin, int yMax, int extendPixels, const cv::Scalar& color, int thickness) {
    const int extendedYMin = std::max(0, yMin - extendPixels);
    drawLineForYRange(image, model, extendedYMin, yMax, color, thickness);
}

bool isHeadless() {
    const char* display = std::getenv("DISPLAY");
    return display == nullptr || std::string(display).empty();
}

} // namespace

// Convert standard Hough (rho, theta) to slope-intercept; returns nullopt if invalid
static std::optional<LineModel> lineModelFromPolar(float rho, float theta) {
    const double sinT = std::sin(theta);
    const double cosT = std::cos(theta);
    if (std::abs(sinT) < 1e-6) {
        return std::nullopt; // vertical line; avoid division by zero here
    }
    LineModel m;
    m.slope = -cosT / sinT; // -cot(theta)
    m.intercept = rho / sinT;
    return m;
}

static std::optional<LineModel> fallbackStandardHough(const cv::Mat& closedEdges, bool wantLeft, int imageWidth, int imageHeight) {
    const int midX = imageWidth / 2;
    cv::Rect roiRect = wantLeft ? cv::Rect(0, 0, midX, imageHeight) : cv::Rect(midX, 0, imageWidth - midX, imageHeight);
    cv::Mat roi = closedEdges(roiRect);

    std::vector<cv::Vec2f> linesPolar;
    // Lower threshold a bit to pick dashed patterns
    cv::HoughLines(roi, linesPolar, 1, CV_PI / 180.0, 110);
    for (const cv::Vec2f& l : linesPolar) {
        const float rho = l[0];
        const float theta = l[1];
        auto maybe = lineModelFromPolar(rho + 0.0f, theta);
        if (!maybe) continue;
        LineModel model = *maybe;
        // Adjust intercept due to ROI offset
        // For y = m x + b, shifting x' = x + roi.x -> y = m x' + (b - m*roi.x)
        model.intercept = model.intercept - model.slope * roiRect.x;

        // Filter by slope direction
        if (wantLeft && model.slope >= -0.15) continue;   // expect negative slope
        if (!wantLeft && model.slope <= 0.15) continue;   // expect positive slope

        // Bottom intersection check
        const int yRef = static_cast<int>(0.75 * imageHeight);
        const double xAtRef = (yRef - model.intercept) / model.slope;
        if (wantLeft) {
            if (!(xAtRef > 0.05 * imageWidth && xAtRef < 0.55 * imageWidth)) continue;
        } else {
            if (!(xAtRef > 0.45 * imageWidth && xAtRef < 0.95 * imageWidth)) continue;
        }
        return model;
    }
    return std::nullopt;
}

static std::optional<LineModel> fallbackFitLineFromPoints(const cv::Mat& closedEdges, bool wantLeft, int imageWidth, int imageHeight) {
    const int midX = imageWidth / 2;
    // Collect non-zero points
    std::vector<cv::Point> nonZero;
    cv::findNonZero(closedEdges, nonZero);
    if (nonZero.empty()) return std::nullopt;

    std::vector<cv::Point2f> candidates;
    candidates.reserve(nonZero.size());
    for (const cv::Point& p : nonZero) {
        if (wantLeft && p.x >= midX) continue;
        if (!wantLeft && p.x <= midX) continue;
        candidates.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    if (candidates.size() < 50) return std::nullopt; // not enough signal

    // Optionally subsample to limit computation
    if (candidates.size() > 5000) {
        std::vector<cv::Point2f> reduced;
        reduced.reserve(5000);
        const size_t step = candidates.size() / 5000;
        for (size_t i = 0; i < candidates.size(); i += step) {
            reduced.push_back(candidates[i]);
        }
        candidates.swap(reduced);
    }

    cv::Vec4f lineParams;
    cv::fitLine(candidates, lineParams, cv::DIST_HUBER, 0, 0.01, 0.01);
    const double vx = static_cast<double>(lineParams[0]);
    const double vy = static_cast<double>(lineParams[1]);
    const double x0 = static_cast<double>(lineParams[2]);
    const double y0 = static_cast<double>(lineParams[3]);
    if (std::abs(vx) < 1e-6) return std::nullopt; // near vertical

    LineModel model;
    model.slope = vy / vx;
    model.intercept = y0 - model.slope * x0;

    // Enforce direction and reasonable slope magnitude
    if (wantLeft && model.slope >= -0.2) return std::nullopt;
    if (!wantLeft && model.slope <= 0.2) return std::nullopt;
    if (std::abs(model.slope) < 0.3) return std::nullopt;

    // Bottom intersection sanity
    const double xBottom = (imageHeight - model.intercept) / model.slope;
    if (wantLeft) {
        if (!(xBottom > 0.05 * imageWidth && xBottom < 0.55 * imageWidth)) return std::nullopt;
    } else {
        if (!(xBottom > 0.45 * imageWidth && xBottom < 0.95 * imageWidth)) return std::nullopt;
    }
    return model;
}

int main(int argc, char** argv) {
    // Default input folder relative to the build directory
    std::string dataFolder = "../data";
    if (argc >= 2) {
        dataFolder = argv[1];
    }

    if (!fs::exists(dataFolder)) {
        std::cerr << "Input folder does not exist: " << dataFolder << std::endl;
        return 1;
    }

    const bool headless = isHeadless();
    const bool debugEnabled = [](){
        const char* dbg = std::getenv("DEBUG_LANES");
        return dbg != nullptr && std::string(dbg).size() > 0 && std::string(dbg) != "0";
    }();

    for (const auto& entry : fs::directory_iterator(dataFolder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path path = entry.path();
        const std::string ext = path.extension().string();
        if (!(ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".PNG" || ext == ".JPG" || ext == ".JPEG")) {
            continue; // skip non-image files
        }
        const std::string stem = path.stem().string();
        auto endsWith = [](const std::string& s, const std::string& suffix){
            return s.size() >= suffix.size() && s.rfind(suffix) == s.size() - suffix.size();
        };
        if (endsWith(stem, "_lanes") || endsWith(stem, "_edges") || endsWith(stem, "_mask") || endsWith(stem, "_roi") || endsWith(stem, "_closed") || endsWith(stem, "_overlay")) {
            continue; // skip debug/processed outputs
        }

        const std::string imgPath = path.string();
        cv::Mat original = cv::imread(imgPath);
        if (original.empty()) {
            // Not an image or failed to open; skip
            continue;
        }

        // 1) Grayscale
        cv::Mat gray;
        cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);

        // 2) Blur to reduce noise
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

        // 3) Canny edges
        cv::Mat edges;
        cv::Canny(blurred, edges, 50, 150);

    // 3.5) Color mask to emphasize lane paint (white/yellow)
    cv::Mat colorMask = buildLaneColorMask(original);
    cv::Mat edgesColorFiltered;
    cv::bitwise_and(edges, colorMask, edgesColorFiltered);

        // 4) ROI mask (focus on road area)
        cv::Mat maskedEdges = applyRegionOfInterestMask(edgesColorFiltered);

    // 4.5) Morphological close to connect dashed segments
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat closed;
    cv::morphologyEx(maskedEdges, closed, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(closed, closed, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        // 5) Probabilistic Hough Transform
    std::vector<cv::Vec4i> houghLines;
    cv::HoughLinesP(
        closed,
        houghLines,
        1.0,
        CV_PI / 180.0,
        28,                                   // slightly lower threshold
        std::max(25.0, original.cols * 0.05), // shorter minLineLength
        45.0                                  // larger maxLineGap
    );

        // Separate into left and right lane candidates using bottom-intersection constraint
        const int yRef = static_cast<int>(0.75 * original.rows); // classify using a y within ROI to avoid car bottom
        ClassifiedLines classified = classifyAndFilterLines(houghLines, original.cols, original.rows, yRef);
        std::vector<cv::Vec4i> leftCandidates = std::move(classified.left);
        std::vector<cv::Vec4i> rightCandidates = std::move(classified.right);

        // 6) Fit averaged left/right lines
        std::optional<LineModel> leftModel = fitAverageLine(leftCandidates, /*wantNegativeSlope=*/true);
        std::optional<LineModel> rightModel = fitAverageLine(rightCandidates, /*wantNegativeSlope=*/false);

        // Fallback: If left or right missing (e.g., dashed), try standard Hough on respective ROI half
        if (!leftModel) {
            leftModel = fallbackStandardHough(closed, /*wantLeft=*/true, original.cols, original.rows);
            if (!leftModel) {
                leftModel = fallbackFitLineFromPoints(closed, /*wantLeft=*/true, original.cols, original.rows);
            }
        }
        if (!rightModel) {
            rightModel = fallbackStandardHough(closed, /*wantLeft=*/false, original.cols, original.rows);
            if (!rightModel) {
                rightModel = fallbackFitLineFromPoints(closed, /*wantLeft=*/false, original.cols, original.rows);
            }
        }

        // 7) Draw overlay
        cv::Mat output = original.clone();

        // Optionally draw all Hough segments (light color)
        for (const cv::Vec4i& l : houghLines) {
            cv::line(output, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }

        const int yMin = static_cast<int>(0.62 * original.rows);
        const int yMax = original.rows;

        if (leftModel) {
            drawExtendedLine(output, *leftModel, yMin, yMax, 60, cv::Scalar(0, 255, 0), 4);
        }
        if (rightModel) {
            drawExtendedLine(output, *rightModel, yMin, yMax, 60, cv::Scalar(0, 255, 0), 4);
        }

        // Debug artifacts if enabled
        if (debugEnabled) {
            cv::imwrite((path.parent_path() / (path.stem().string() + std::string("_edges.png"))).string(), edges);
            cv::imwrite((path.parent_path() / (path.stem().string() + std::string("_mask.png"))).string(), colorMask);
            cv::imwrite((path.parent_path() / (path.stem().string() + std::string("_roi.png"))).string(), maskedEdges);
            cv::imwrite((path.parent_path() / (path.stem().string() + std::string("_closed.png"))).string(), closed);
        }

        // 8) Save and optionally show
        const std::string outPath = (path.parent_path() / (path.stem().string() + std::string("_lanes.png"))).string();
        cv::imwrite(outPath, output);

        if (!headless) {
            cv::imshow("Lane Detection", output);
            cv::waitKey(500);
        }
    }

    if (!headless) {
    cv::destroyAllWindows();
    }
    return 0;
}
