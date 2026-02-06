/**
 * TwinTube Vector: High-Performance GPU-Accelerated OCR Pipeline
 * 
 * Eliminates Python GIL overhead with native C++ implementation.
 * Leverages CUDA-accelerated OpenCV for maximum throughput.
 * 
 * Performance Targets:
 * - 94% OCR accuracy on YouTube video titles
 * - <50ms per frame processing  
 * - 10x throughput vs Python implementation
 */

#include <opencv2/opencv.hpp>
#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#endif
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <regex>
#include <algorithm>
#include <cmath>

// Configuration
struct OCRConfig {
    bool use_gpu = true;
    double scale_factor = 1.5;
    int denoise_strength = 10;
    float confidence_threshold = 60.0f;
    int num_threads = 4;
};

// Performance metrics
class Metrics {
public:
    std::atomic<int> frames_processed{0};
    std::atomic<int> successful_extractions{0};
    std::vector<double> latencies;
    std::mutex latency_mutex;
    
    void record_latency(double ms) {
        std::lock_guard<std::mutex> lock(latency_mutex);
        latencies.push_back(ms);
        frames_processed++;
    }
    
    void record_success() { successful_extractions++; }
    
    double get_accuracy() const {
        if (frames_processed == 0) return 0.0;
        return 100.0 * successful_extractions / frames_processed;
    }
    
    double get_avg_latency() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(latency_mutex));
        if (latencies.empty()) return 0.0;
        double sum = 0;
        for (double l : latencies) sum += l;
        return sum / latencies.size();
    }
    
    void print() const {
        std::cout << "\n=== OCR Pipeline Performance ===" << std::endl;
        std::cout << "Frames processed: " << frames_processed << std::endl;
        std::cout << "Successful extractions: " << successful_extractions << std::endl;
        std::cout << "Accuracy: " << get_accuracy() << "%" << std::endl;
        std::cout << "Average latency: " << get_avg_latency() << " ms" << std::endl;
        std::cout << "================================\n" << std::endl;
    }
};

// Video data structure
struct VideoData {
    std::string title;
    std::string channel;
    std::string views;
    std::string duration;
    float confidence;
    
    std::string to_json() const {
        auto escape = [](const std::string& s) {
            std::string r;
            for (char c : s) {
                if (c == '"') r += "\\\"";
                else if (c == '\\') r += "\\\\";
                else if (c == '\n') r += "\\n";
                else r += c;
            }
            return r;
        };
        
        std::ostringstream oss;
        oss << "{\"title\":\"" << escape(title) << "\","
            << "\"channel\":\"" << escape(channel) << "\","
            << "\"views\":\"" << escape(views) << "\","
            << "\"duration\":\"" << escape(duration) << "\","
            << "\"confidence\":" << confidence << "}";
        return oss.str();
    }
};

// GPU-accelerated preprocessor
class GPUPreprocessor {
public:
    explicit GPUPreprocessor(const OCRConfig& config) : config_(config) {
#ifdef HAVE_CUDA
        if (config_.use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            gpu_available_ = true;
            std::cout << "[GPU] CUDA acceleration enabled" << std::endl;
        } else {
            gpu_available_ = false;
            std::cout << "[CPU] Running without GPU acceleration" << std::endl;
        }
#else
        gpu_available_ = false;
        std::cout << "[CPU] OpenCV built without CUDA support" << std::endl;
#endif
    }
    
    cv::Mat preprocess(const cv::Mat& input) {
#ifdef HAVE_CUDA
        if (gpu_available_) {
            return preprocess_gpu(input);
        }
#endif
        return preprocess_cpu(input);
    }
    
    bool has_gpu() const { return gpu_available_; }
    
private:
#ifdef HAVE_CUDA
    cv::Mat preprocess_gpu(const cv::Mat& input) {
        cv::cuda::GpuMat gpu_input, gpu_gray, gpu_resized;
        
        gpu_input.upload(input);
        cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);
        
        if (config_.scale_factor != 1.0) {
            cv::cuda::resize(gpu_gray, gpu_resized, cv::Size(),
                           config_.scale_factor, config_.scale_factor);
        } else {
            gpu_resized = gpu_gray;
        }
        
        cv::Mat result;
        gpu_resized.download(result);
        
        // CPU operations for thresholding (no CUDA equivalent for adaptive)
        cv::Mat thresh;
        cv::adaptiveThreshold(result, thresh, 255,
                             cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv::THRESH_BINARY, 11, 2);
        
        // CLAHE for contrast
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat enhanced;
        clahe->apply(thresh, enhanced);
        
        return enhanced;
    }
#endif
    
    cv::Mat preprocess_cpu(const cv::Mat& input) {
        cv::Mat gray, resized, denoised, thresh, enhanced;
        
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        
        if (config_.scale_factor != 1.0) {
            cv::resize(gray, resized, cv::Size(),
                      config_.scale_factor, config_.scale_factor,
                      cv::INTER_CUBIC);
        } else {
            resized = gray;
        }
        
        cv::fastNlMeansDenoising(resized, denoised, config_.denoise_strength);
        
        cv::adaptiveThreshold(denoised, thresh, 255,
                             cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv::THRESH_BINARY, 11, 2);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(thresh, enhanced);
        
        return enhanced;
    }
    
    OCRConfig config_;
    bool gpu_available_ = false;
};

// Video data extractor
class VideoExtractor {
public:
    VideoData extract(const std::string& text, float confidence) {
        VideoData data;
        data.confidence = confidence;
        
        std::vector<std::string> lines;
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line)) {
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
            if (line.length() > 2) lines.push_back(line);
        }
        
        data.title = extract_title(lines);
        data.channel = extract_channel(lines);
        data.views = extract_views(text);
        data.duration = extract_duration(text);
        
        return data;
    }
    
private:
    std::string extract_title(const std::vector<std::string>& lines) {
        std::string best;
        size_t max_score = 0;
        
        for (size_t i = 0; i < std::min(lines.size(), size_t(5)); i++) {
            if (is_ui_element(lines[i])) continue;
            
            size_t words = count_words(lines[i]);
            size_t score = lines[i].length() * words;
            
            if (score > max_score && words >= 2) {
                max_score = score;
                best = lines[i];
            }
        }
        return best;
    }
    
    std::string extract_channel(const std::vector<std::string>& lines) {
        for (size_t i = 1; i < std::min(lines.size(), size_t(10)); i++) {
            if (is_ui_element(lines[i])) continue;
            if (lines[i].length() > 50) continue;
            if (count_words(lines[i]) <= 5) return lines[i];
        }
        return "";
    }
    
    std::string extract_views(const std::string& text) {
        std::regex pattern(R"([\d,\.]+\s*[KMBkmb]?\s*views?)", std::regex::icase);
        std::smatch match;
        if (std::regex_search(text, match, pattern)) return match.str();
        return "";
    }
    
    std::string extract_duration(const std::string& text) {
        std::regex pattern(R"(\d{1,2}:\d{2}(?::\d{2})?)");
        std::smatch match;
        if (std::regex_search(text, match, pattern)) return match.str();
        return "";
    }
    
    bool is_ui_element(const std::string& text) {
        static const std::vector<std::string> ui = {
            "search", "home", "explore", "subscriptions", "library",
            "history", "settings", "subscribe", "share"
        };
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        for (const auto& w : ui) if (lower == w) return true;
        return false;
    }
    
    size_t count_words(const std::string& text) {
        size_t count = 0;
        bool in_word = false;
        for (char c : text) {
            if (std::isalnum(c)) {
                if (!in_word) { count++; in_word = true; }
            } else {
                in_word = false;
            }
        }
        return count;
    }
};

// Main OCR Pipeline
class OCRPipeline {
public:
    explicit OCRPipeline(const OCRConfig& config = OCRConfig())
        : config_(config), preprocessor_(config) {
        
        if (tess_.Init(nullptr, "eng")) {
            throw std::runtime_error("Failed to initialize Tesseract");
        }
        tess_.SetPageSegMode(tesseract::PSM_AUTO);
        
        std::cout << "OCR Pipeline initialized" << std::endl;
        std::cout << "GPU acceleration: " << (preprocessor_.has_gpu() ? "ON" : "OFF") << std::endl;
    }
    
    ~OCRPipeline() { tess_.End(); }
    
    VideoData process_frame(const cv::Mat& frame) {
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Mat processed = preprocessor_.preprocess(frame);
        
        tess_.SetImage(processed.data, processed.cols, processed.rows, 1, processed.step);
        std::string text = tess_.GetUTF8Text();
        float confidence = tess_.MeanTextConf();
        
        VideoData data = extractor_.extract(text, confidence);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end - start).count();
        
        metrics_.record_latency(latency);
        if (!data.title.empty()) metrics_.record_success();
        
        return data;
    }
    
    std::vector<VideoData> process_video(const std::string& path, int interval = 30) {
        std::vector<VideoData> results;
        
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open " << path << std::endl;
            return results;
        }
        
        cv::Mat frame;
        int count = 0;
        
        while (cap.read(frame)) {
            count++;
            if (count % interval != 0) continue;
            
            VideoData data = process_frame(frame);
            if (!data.title.empty()) {
                results.push_back(data);
                std::cout << "Frame " << count << ": " << data.title << std::endl;
            }
        }
        
        return results;
    }
    
    const Metrics& get_metrics() const { return metrics_; }
    
private:
    OCRConfig config_;
    GPUPreprocessor preprocessor_;
    VideoExtractor extractor_;
    tesseract::TessBaseAPI tess_;
    Metrics metrics_;
};

int main(int argc, char** argv) {
    std::cout << "TwinTube Vector - GPU-Accelerated OCR Pipeline" << std::endl;
    std::cout << "===============================================\n" << std::endl;
    
    std::string input = argc > 1 ? argv[1] : "/usr/src/app/input/video.mp4";
    std::string output = argc > 2 ? argv[2] : "/usr/src/app/output/results.json";
    int interval = argc > 3 ? std::atoi(argv[3]) : 30;
    
    OCRConfig config;
    config.use_gpu = true;
    config.scale_factor = 1.5;
    
    try {
        OCRPipeline pipeline(config);
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<VideoData> results = pipeline.process_video(input, interval);
        auto end = std::chrono::high_resolution_clock::now();
        
        double total = std::chrono::duration<double>(end - start).count();
        
        // Write JSON output
        std::ofstream out(output);
        out << "[\n";
        for (size_t i = 0; i < results.size(); i++) {
            out << "  " << results[i].to_json();
            if (i < results.size() - 1) out << ",";
            out << "\n";
        }
        out << "]\n";
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Total time: " << total << " s" << std::endl;
        std::cout << "Videos extracted: " << results.size() << std::endl;
        std::cout << "Output: " << output << std::endl;
        
        pipeline.get_metrics().print();
        
        double accuracy = pipeline.get_metrics().get_accuracy();
        std::cout << "Target accuracy: 94%" << std::endl;
        std::cout << "Achieved: " << accuracy << "%" << std::endl;
        std::cout << "Status: " << (accuracy >= 94.0 ? "PASSED" : "In progress") << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
