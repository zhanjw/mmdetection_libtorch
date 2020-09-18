//
// Created by dl on 2020/1/2.
//

#ifndef DETECTOR_TYPES_HPP
#define DETECTOR_TYPES_HPP
#include <vector>
#include <string>
#include<opencv2/opencv.hpp>



struct DetectedBox {
    cv::Rect box;
    int label;
    float score;

    DetectedBox()
    {
        score = 0;
        label = -1;
    }
};


enum class DetetorType : int
{
    SSD = 0,
    Retinanet = 1,
    FasterRcnn = 2,
};


struct AnchorHeadParams {
    int nms_pre_;
    int use_sigmoid_;
    float nms_thresh_;
    float score_thresh_;
    int max_per_img_;
    std::vector<int> anchor_strides_;
    std::vector<float> target_means_;
    std::vector<float> target_stds_;
    std::vector<float> bbox_target_means_;
    std::vector<float> bbox_target_stds_;
};

struct RetinaHeadParams {
    int octave_base_scale_;
    int scales_per_octave_;
    std::vector<float> anchor_ratios_;
};


struct SSDHeadParams {
    int input_size_;
    std::vector<float> basesize_ratio_range_;
    std::vector<std::vector<int>> anchor_ratios_;
};

struct RPNHeadParams {
    int class_num_;
    int nms_across_levels_;
    int nms_post_;
    int max_num_;
    int min_bbox_size_;
    std::vector<float> anchor_scales_;
    std::vector<float> anchor_ratios_;
};

struct TransformParams {
    std::vector<float> mean_;
    std::vector<float> std_;
    int to_rgb_;
    std::vector<int> img_scale_;
    int keep_ratio_;
    int pad_;
    std::vector<int> img_shape_;
    float scale_factor_;
    std::vector<float> vector_scale_factor_;
};

struct RoiExtractorParams {
    std::string type_;
    int out_size_;
    int sample_num_;
    int out_channels_;
    std::vector<int> featmap_strides_;
};

struct FPNParams {
    int out_channels_;
    int num_outs_;
};

struct Params {
public:
    Params() = default;
    ~Params() = default;

    DetetorType detector_type_;
    std::string model_path_;
    std::string save_path_;
    float conf_thresh_;

    TransformParams transform_params_;
    AnchorHeadParams anchor_head_params_;


    SSDHeadParams ssd_head_params_;
    RetinaHeadParams retina_head_params_;
    RPNHeadParams rpn_head_params_;
    RoiExtractorParams roi_extractor_params_;
    FPNParams fpn_params_;


    int Read(const std::string &config_file);
};
#endif //DETECTOR_TYPES_HPP
