#include <iostream>
#include <boost/filesystem.hpp>
#include "opencv2/opencv.hpp"
#include "cv_dnn_ultraface.h"
#include "opencv2/imgproc.hpp"

cv::Mat process_image(const std::string& image_path, UltraFace model) {
    cv::Mat image = cv::imread(image_path);
    std::vector<FaceInfo> boxes;
    cv::TickMeter tm;

    tm.reset();
    tm.start();
    model.detect(image, boxes);
    tm.stop();
    std::cout << "inference time: " << tm.getTimeMilli() << " ms\n";
    for (FaceInfo &box : boxes) {
        std::cout << box.score << ": [" << box.x1 << ", " << box.x2 << ", " << box.y1 << ", " << box.y2 << "]\n";
        cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), {0,0,255}, 2);
        cv::putText(image, std::to_string(box.score), cv::Point(box.x1, box.y1+10), 1, 1, {255,255,255});
    }

    std::string winname = "img";
    cv::namedWindow(winname);
    cv::imshow(winname, image);
    cv::waitKey(0);
    cv::destroyWindow(winname);
    return image;
}

int main() {
    std::string model_path = "../models/version-RFB-320_without_postprocessing.onnx";
    std::cout << "Model path: " << model_path << std::endl;

    std::string image_dir = "/home/ppotrykus/Datasets/image/face_thermo/fever_detector/resized/";
//    For best performance make sure images are in 480x360 resolution.
    UltraFace model = {model_path, 480, 360};
    cv::Mat detection;
    for (boost::filesystem::directory_entry& image : boost::filesystem::directory_iterator(image_dir)){
        detection = process_image(image.path().string(), model);
    }
//    cv::imwrite("detection.jpg", detection);
    return 0;
}
