/**
 * ocr_preprocess — C++ image preprocessor for the OCR pipeline.
 *
 * Takes a screenshot image, detects text regions with OpenCV
 * (MSER + morphology + contours), crops and cleans each region,
 * and writes:
 *   • one PNG per region   → <outdir>/region_NNN.png
 *   • a JSON manifest      → <outdir>/manifest.json
 *
 * Usage:
 *   ocr_preprocess <input_image> <output_dir>
 *
 * The manifest is an array of objects:
 *   { "file": "region_000.png", "x":…, "y":…, "w":…, "h":… }
 *
 * These crops are already binarised and cleaned — ready for Tesseract.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// ─────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────

struct TextRegion {
    int x, y, w, h;
    std::string file;
};

static std::string escape_json(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"')       r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else                r += c;
    }
    return r;
}

static std::string regions_to_json(const std::vector<TextRegion>& regions) {
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < regions.size(); ++i) {
        auto& r = regions[i];
        oss << "  {\"file\":\"" << escape_json(r.file) << "\","
            << "\"x\":" << r.x << ","
            << "\"y\":" << r.y << ","
            << "\"w\":" << r.w << ","
            << "\"h\":" << r.h << "}";
        if (i + 1 < regions.size()) oss << ",";
        oss << "\n";
    }
    oss << "]\n";
    return oss.str();
}

// ─────────────────────────────────────────────────────
//  Preprocessing: produce a clean binary image for OCR
// ─────────────────────────────────────────────────────

static cv::Mat clean_for_ocr(const cv::Mat& bgr_crop) {
    cv::Mat gray;
    cv::cvtColor(bgr_crop, gray, cv::COLOR_BGR2GRAY);

    // Upscale small regions so Tesseract has enough pixels to work with
    if (gray.cols < 300 || gray.rows < 40) {
        double sf = std::max(300.0 / gray.cols, 40.0 / gray.rows);
        sf = std::min(sf, 3.0);  // cap at 3×
        cv::resize(gray, gray, cv::Size(), sf, sf, cv::INTER_CUBIC);
    }

    // Adaptive threshold: handles both dark-on-light and light-on-dark
    cv::Mat bin;
    cv::adaptiveThreshold(gray, bin, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 31, 15);

    // Decide polarity: if the majority of the crop is dark, invert so
    // text is always black-on-white for Tesseract.
    double mean_val = cv::mean(gray)[0];
    if (mean_val < 127) {
        cv::bitwise_not(bin, bin);
    }

    // Light morphological close to bridge broken characters
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

    return bin;
}

// ─────────────────────────────────────────────────────
//  Text region detection
// ─────────────────────────────────────────────────────

/**
 * Detect text-line bounding boxes using a two-pass approach:
 *   Pass 1 — MSER (Maximally Stable Extremal Regions) to find
 *            character-level blobs.
 *   Pass 2 — Dilate the blob mask horizontally to merge characters
 *            into line-level bounding boxes, then find contours.
 *
 * Returns bounding rects sorted top-to-bottom.
 */
static std::vector<cv::Rect> detect_text_regions(const cv::Mat& bgr,
                                                  int min_area  = 80,
                                                  int max_area  = 0,
                                                  int pad       = 6) {
    int img_h = bgr.rows;
    int img_w = bgr.cols;
    if (max_area <= 0) max_area = img_h * img_w / 4;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // --- Pass 1: MSER character blobs ---
    auto mser = cv::MSER::create(
        /* delta            */ 5,
        /* min_area         */ 30,
        /* max_area         */ max_area,
        /* max_variation    */ 0.25,
        /* min_diversity    */ 0.2,
        /* max_evolution    */ 200,
        /* area_threshold   */ 1.01,
        /* min_margin       */ 0.003,
        /* edge_blur_size   */ 5
    );

    std::vector<std::vector<cv::Point>> regions;
    std::vector<cv::Rect> mser_boxes;
    mser->detectRegions(gray, regions, mser_boxes);

    // Paint detected MSER blobs onto a mask
    cv::Mat mask = cv::Mat::zeros(img_h, img_w, CV_8UC1);
    for (auto& box : mser_boxes) {
        // Sanity filter: text characters are not huge blobs
        if (box.area() < min_area || box.area() > max_area) continue;
        float aspect = static_cast<float>(box.width) / std::max(1, box.height);
        if (aspect > 15 || aspect < 0.05) continue;  // skip extreme shapes
        cv::rectangle(mask, box, cv::Scalar(255), cv::FILLED);
    }

    // --- Pass 2: horizontal dilation to merge characters into words/lines ---
    // Keep kernel small: only bridge intra-word gaps (~10-25px),
    // NOT the ~200px gap between adjacent video card columns.
    int kw = std::max(10, img_w / 120);   // ~0.8% of image width (~25px on 3024)
    int kh = 3;
    cv::Mat h_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kw, kh));
    cv::dilate(mask, mask, h_kernel, cv::Point(-1, -1), 1);  // single iteration

    // Tiny vertical dilation to merge descenders/ascenders within one text line
    cv::Mat v_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
    cv::dilate(mask, mask, v_kernel, cv::Point(-1, -1), 1);

    // Find contours on the merged mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Convert to bounding rects, filter, and pad
    std::vector<cv::Rect> result;
    for (auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);

        // Filter: skip tiny fragments and page-spanning boxes
        if (r.width < 30 || r.height < 10) continue;
        if (r.width > img_w * 0.85 && r.height > img_h * 0.5) continue;

        // Skip regions wider than 40% of image — a single video card
        // title never spans more than ~1/3 of a YouTube grid row.
        if (r.width > img_w * 0.40) continue;

        // Skip regions taller than 20% of image — likely thumbnails.
        if (r.height > img_h * 0.20) continue;

        // Pad the box a few pixels
        r.x = std::max(0, r.x - pad);
        r.y = std::max(0, r.y - pad);
        r.width  = std::min(img_w - r.x, r.width  + 2 * pad);
        r.height = std::min(img_h - r.y, r.height + 2 * pad);

        result.push_back(r);
    }

    // Sort top-to-bottom, then left-to-right
    std::sort(result.begin(), result.end(), [](const cv::Rect& a, const cv::Rect& b) {
        if (std::abs(a.y - b.y) > 20) return a.y < b.y;
        return a.x < b.x;
    });

    return result;
}

// ─────────────────────────────────────────────────────
//  Merge overlapping / nearly-overlapping boxes
// ─────────────────────────────────────────────────────

static std::vector<cv::Rect> merge_overlapping(std::vector<cv::Rect>& boxes,
                                                float overlap_thresh = 0.3) {
    if (boxes.empty()) return boxes;

    // Sort by area descending so larger boxes absorb smaller ones
    std::sort(boxes.begin(), boxes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() > b.area();
    });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<cv::Rect> result;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        cv::Rect merged = boxes[i];

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;

            cv::Rect inter = merged & boxes[j];
            if (inter.area() == 0) continue;

            float iou = static_cast<float>(inter.area()) /
                        std::min(merged.area(), boxes[j].area());
            if (iou > overlap_thresh) {
                merged |= boxes[j];  // union
                suppressed[j] = true;
            }
        }
        result.push_back(merged);
    }
    return result;
}

// ─────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ocr_preprocess <input_image> <output_dir>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_dir = argv[2];

    cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: cannot read image " << input_path << std::endl;
        return 1;
    }

    // Ignore top 100px of browser chrome / nav bar
    int crop_top = std::min(100, image.rows / 8);
    cv::Mat working = image(cv::Rect(0, crop_top, image.cols, image.rows - crop_top));

    // Detect text regions
    auto boxes = detect_text_regions(working);
    boxes = merge_overlapping(boxes);

    std::cerr << "[C++ preprocess] Detected " << boxes.size() << " text regions" << std::endl;

    // Crop, clean, and write each region
    std::vector<TextRegion> manifest;
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::Rect& r = boxes[i];
        cv::Mat crop = working(r);
        cv::Mat cleaned = clean_for_ocr(crop);

        char fname[64];
        std::snprintf(fname, sizeof(fname), "region_%03zu.png", i);
        std::string out_path = output_dir + "/" + fname;
        cv::imwrite(out_path, cleaned);

        TextRegion tr;
        tr.file = fname;
        tr.x = r.x;
        tr.y = r.y + crop_top;   // map back to original image coords
        tr.w = r.width;
        tr.h = r.height;
        manifest.push_back(tr);
    }

    // Write manifest
    std::string manifest_path = output_dir + "/manifest.json";
    std::ofstream mf(manifest_path);
    mf << regions_to_json(manifest);
    mf.close();

    std::cout << manifest_path << std::endl;
    return 0;
}
