#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <fstream>
#include <iostream>

int main() {
    std::string videoPath = "/usr/src/app/input/video.mp4";
    std::string outputPath = "/usr/src/app/output/titles.txt";

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }

    tesseract::TessBaseAPI tess;
    if (tess.Init(NULL, "eng")) {
        std::cerr << "Error initializing tesseract." << std::endl;
        return 1;
    }

    std::ofstream outputFile(outputPath);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }

    cv::Mat frame;
    int frameCount = 0;
    while (cap.read(frame)) {
        frameCount++;
        if (frameCount % 30 != 0) continue; // process every 30th frame for efficiency

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, gray, 150, 255, cv::THRESH_BINARY);

        tess.SetImage(gray.data, gray.cols, gray.rows, 1, gray.step);
        std::string text = tess.GetUTF8Text();
        if (text.find("YouTube") != std::string::npos || text.length() > 5) {
            outputFile << "Frame " << frameCount << ": " << text << std::endl;
        }
    }

    std::cout << "Processing complete. Results saved to " << outputPath << std::endl;
    return 0;
}
