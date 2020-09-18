#include "types.hpp"
#include "Detector.hpp"

int main() {

//    std::string config_file = "../config/config_ssd.json";
    std::string config_file = "../config/config_re.json";
//    std::string config_file = "../config/config_faster_rcnn.json";
    Params params;
    long res = params.Read(config_file);
    if (res == -1) {
        return -1;
    }


    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    Detector detector;
    res = detector.Create(params.detector_type_);
    if (res < 0) {
        return -1;
    }
    detector.LoadParams( params, &device_type);
    detector.LoadTracedModule();

    cv::Mat image = cv::imread("../image/demo.jpg");
    std::vector<DetectedBox> detected_boxes;
    res = detector.Detect(image, detected_boxes);
    std::cout << "detect "<<detected_boxes.size() <<" bboxs"<< std::endl;
    if (res == 0) {
        for(u_int64_t i = 0; i < detected_boxes.size(); i++) {
            /*
            std::cout << detected_boxes[i].box.x <<" ";
            std::cout << detected_boxes[i].box.y << " ";
            std::cout << detected_boxes[i].box.width << " ";
            std::cout << detected_boxes[i].box.height <<std::endl;
             */
            cv::rectangle(image, detected_boxes[i].box, cv::Scalar(0, 0, 255), 1, 1, 0);
        }
        cv::imwrite("../image/result.jpg", image);
    }

    return 0;
}

