#include <iostream>
#include <boost/filesystem.hpp>
#include "opencv2/opencv.hpp"
#include "cv_dnn_ultraface.h"
#include "opencv2/imgproc.hpp"

cv::Mat process_image(const std::string& image_path, UltraFace model, bool show=true) {
    cv::Mat image = cv::imread(image_path);
    std::vector<FaceInfo> boxes;
    cv::TickMeter tm;

    tm.reset();
    tm.start();
    model.detect(image, boxes);
    tm.stop();
    std::cout << "inference time: " << tm.getTimeMilli() << " ms\n";
    for (FaceInfo &box : boxes) {
        std::cout << box.score << ": [" << box.x1 << ", " << box.x2 << ", " << box.y1 << ", " << box.y2 << "] @ " << image_path << "\n";
        cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), {0,0,255}, 2);
        cv::putText(image, std::to_string(box.score), cv::Point(box.x1, box.y1+10), 1, 1, {255,255,255});
    }

    if (show) {
        std::string winname = image_path;
        cv::namedWindow(winname);
        cv::imshow(winname, image);
        cv::waitKey(0);
        cv::destroyWindow(winname);
    }
    return image;
}

int main(int argc, char* argv[]) {
    std::string model_path = "../models/version-RFB-640_without_postprocessing.onnx";
//    std::string model_path = "../models/version-RFB-320_without_postprocessing.onnx";
    std::string image_dir = argv[1];
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Image dir: " << argv[1] << std::endl;

    UltraFace model = {model_path, 640, 480, 6, 0.6};
    cv::Mat detection;
    for (boost::filesystem::directory_entry& image : boost::filesystem::recursive_directory_iterator(image_dir)){
        if (boost::filesystem::is_regular_file(image)) {
//            std::cout << image << std::endl;
            detection = process_image(image.path().string(), model, false);
//            std::string out_path = "/home/ppotrykus/Datasets/image/face_thermo/.detections/vid1/"+image.path().filename().string();
//            cv::imwrite(out_path, detection);
        }
    }
    return 0;
}
