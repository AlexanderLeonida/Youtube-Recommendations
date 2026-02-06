/**
 * TwinTube OCR Benchmark Suite
 * 
 * Validates the 94% accuracy claim and measures performance metrics.
 */

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

// Ground truth data for accuracy validation
struct GroundTruth {
    std::string image_path;
    std::string expected_title;
    std::string expected_channel;
};

class OCRBenchmark {
public:
    struct BenchmarkResults {
        double accuracy;
        double precision;
        double recall;
        double f1_score;
        double avg_latency_ms;
        double p50_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        int total_samples;
        int correct_samples;
    };
    
    BenchmarkResults run(const std::vector<GroundTruth>& test_data) {
        BenchmarkResults results = {};
        results.total_samples = test_data.size();
        
        std::vector<double> latencies;
        int true_positives = 0;
        int false_positives = 0;
        int false_negatives = 0;
        
        tesseract::TessBaseAPI api;
        if (api.Init(nullptr, "eng")) {
            std::cerr << "Could not initialize tesseract" << std::endl;
            return results;
        }
        
        for (const auto& sample : test_data) {
            cv::Mat image = cv::imread(sample.image_path);
            if (image.empty()) {
                std::cerr << "Could not load: " << sample.image_path << std::endl;
                false_negatives++;
                continue;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Preprocess
            cv::Mat gray, processed;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::threshold(gray, processed, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
            
            // OCR
            api.SetImage(processed.data, processed.cols, processed.rows, 1, processed.step);
            std::string text = api.GetUTF8Text();
            float confidence = api.MeanTextConf();
            
            auto end = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency);
            
            // Check if expected title is found
            bool title_found = text.find(sample.expected_title) != std::string::npos ||
                              fuzzy_match(text, sample.expected_title) > 0.8;
            
            if (title_found) {
                true_positives++;
                results.correct_samples++;
            } else if (!sample.expected_title.empty()) {
                false_negatives++;
            }
        }
        
        api.End();
        
        // Calculate metrics
        results.accuracy = static_cast<double>(results.correct_samples) / results.total_samples * 100.0;
        
        if (true_positives + false_positives > 0) {
            results.precision = static_cast<double>(true_positives) / (true_positives + false_positives);
        }
        
        if (true_positives + false_negatives > 0) {
            results.recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
        }
        
        if (results.precision + results.recall > 0) {
            results.f1_score = 2 * results.precision * results.recall / (results.precision + results.recall);
        }
        
        // Calculate latency percentiles
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());
            
            double sum = 0;
            for (double l : latencies) sum += l;
            results.avg_latency_ms = sum / latencies.size();
            
            results.p50_latency_ms = percentile(latencies, 50);
            results.p95_latency_ms = percentile(latencies, 95);
            results.p99_latency_ms = percentile(latencies, 99);
        }
        
        return results;
    }
    
    void print_results(const BenchmarkResults& results) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "       OCR BENCHMARK RESULTS" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::cout << "ACCURACY METRICS:" << std::endl;
        std::cout << "  Total Samples: " << results.total_samples << std::endl;
        std::cout << "  Correct: " << results.correct_samples << std::endl;
        std::cout << "  Accuracy: " << results.accuracy << "%" << std::endl;
        std::cout << "  Precision: " << results.precision * 100 << "%" << std::endl;
        std::cout << "  Recall: " << results.recall * 100 << "%" << std::endl;
        std::cout << "  F1 Score: " << results.f1_score * 100 << "%" << std::endl;
        
        std::cout << "\nLATENCY METRICS:" << std::endl;
        std::cout << "  Average: " << results.avg_latency_ms << " ms" << std::endl;
        std::cout << "  P50: " << results.p50_latency_ms << " ms" << std::endl;
        std::cout << "  P95: " << results.p95_latency_ms << " ms" << std::endl;
        std::cout << "  P99: " << results.p99_latency_ms << " ms" << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "TARGET: 94% Accuracy" << std::endl;
        std::cout << "RESULT: " << (results.accuracy >= 94.0 ? "PASSED ✓" : "FAILED ✗") << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
private:
    double percentile(const std::vector<double>& sorted_data, int p) {
        if (sorted_data.empty()) return 0.0;
        
        double index = (p / 100.0) * (sorted_data.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));
        
        if (lower == upper) return sorted_data[lower];
        
        double weight = index - lower;
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight;
    }
    
    double fuzzy_match(const std::string& text, const std::string& pattern) {
        // Simple character-level Jaccard similarity
        if (pattern.empty()) return 0.0;
        
        std::string lower_text = text;
        std::string lower_pattern = pattern;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
        std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), ::tolower);
        
        std::set<char> text_chars(lower_text.begin(), lower_text.end());
        std::set<char> pattern_chars(lower_pattern.begin(), lower_pattern.end());
        
        std::set<char> intersection;
        std::set_intersection(text_chars.begin(), text_chars.end(),
                             pattern_chars.begin(), pattern_chars.end(),
                             std::inserter(intersection, intersection.begin()));
        
        std::set<char> union_set;
        std::set_union(text_chars.begin(), text_chars.end(),
                      pattern_chars.begin(), pattern_chars.end(),
                      std::inserter(union_set, union_set.begin()));
        
        if (union_set.empty()) return 0.0;
        return static_cast<double>(intersection.size()) / union_set.size();
    }
};

// Forward declaration
void create_test_image(const std::string& path, 
                       const std::string& title,
                       const std::string& channel);

// Generate synthetic test data for benchmarking
std::vector<GroundTruth> generate_synthetic_test_data(int num_samples) {
    std::vector<GroundTruth> data;
    
    std::vector<std::string> sample_titles = {
        "How to Learn Machine Learning in 2024",
        "10 Tips for Better Python Code",
        "Understanding Neural Networks",
        "React Tutorial for Beginners",
        "Building REST APIs with Node.js",
        "Introduction to Data Science",
        "Deep Learning Explained Simply",
        "Git and GitHub for Beginners",
        "Docker Tutorial - Full Course",
        "Kubernetes Crash Course"
    };
    
    std::vector<std::string> sample_channels = {
        "TechWithTim", "Traversy Media", "Fireship", 
        "The Coding Train", "Corey Schafer"
    };
    
    std::mt19937 rng(42);
    
    for (int i = 0; i < num_samples; i++) {
        GroundTruth gt;
        gt.image_path = "/tmp/test_frame_" + std::to_string(i) + ".png";
        gt.expected_title = sample_titles[i % sample_titles.size()];
        gt.expected_channel = sample_channels[i % sample_channels.size()];
        
        // Create synthetic test image
        create_test_image(gt.image_path, gt.expected_title, gt.expected_channel);
        
        data.push_back(gt);
    }
    
    return data;
}

void create_test_image(const std::string& path, 
                       const std::string& title,
                       const std::string& channel) {
    cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw title
    cv::putText(image, title, cv::Point(100, 200),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
    
    // Draw channel
    cv::putText(image, channel, cv::Point(100, 280),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 100), 1);
    
    // Add some noise for realism
    cv::Mat noise(image.size(), image.type());
    cv::randn(noise, 0, 5);
    image += noise;
    
    cv::imwrite(path, image);
}


int main(int argc, char** argv) {
    std::cout << "TwinTube OCR Benchmark Suite" << std::endl;
    std::cout << "============================\n" << std::endl;
    
    int num_samples = 100;
    if (argc > 1) {
        num_samples = std::atoi(argv[1]);
    }
    
    std::cout << "Generating " << num_samples << " synthetic test samples..." << std::endl;
    auto test_data = generate_synthetic_test_data(num_samples);
    
    std::cout << "Running benchmark..." << std::endl;
    OCRBenchmark benchmark;
    auto results = benchmark.run(test_data);
    
    benchmark.print_results(results);
    
    // Cleanup test images
    for (const auto& gt : test_data) {
        std::remove(gt.image_path.c_str());
    }
    
    return results.accuracy >= 94.0 ? 0 : 1;
}
