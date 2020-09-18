#ifndef DETERCOTCOMMON_HPP
#define DETERCOTCOMMON_HPP

#include "torch/torch.h"
#include "torch/script.h"
#include "types.hpp"
#include "bbox.hpp"
#include "transforms.hpp"
#include "AnchorGenerator.hpp"
#include <sys/stat.h>

using namespace std::chrono;

class DetectorCommon
{
public:
    DetectorCommon();
    ~DetectorCommon();

    void LoadParams(const Params& params, torch::DeviceType* device_type);
    void LoadTracedModule();
    virtual void DetectOneStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes);
    virtual void DetectTwoStage(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes);

    virtual void get_anchor_generators(const std::vector<int>& anchor_base_sizes,
                                const std::vector<float>& anchor_scales,
                                const std::vector<float>& anchor_ratios);
    //must know the net size
    virtual void get_anchor_boxes();

protected:
    void get_proposals(torch::Tensor& output, const std::vector<int>& img_shape,
                       torch::Tensor& proposals_bboxes, torch::Tensor&  proposals_scores,
                       bool rpn);
    inline bool exists(const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }

private:
    void init_params();


protected:

    torch::DeviceType * device_;

    DetetorType detector_type_;
    std::string model_path_;
    std::string save_path_;
    int net_width_;
    int net_height_;
    float conf_thresh_;

    TransformParams transform_params_;
    AnchorHeadParams anchor_head_params_;
    RPNHeadParams rpn_head_params_;
    RoiExtractorParams roi_extractor_params_;
    FPNParams fpn_params_;

    std::vector<AnchorGenerator> anchor_generators_;
    torch::Tensor mlvl_anchors_;
    std::unique_ptr<torch::jit::script::Module> module_;
};

#endif // DETERCOTCOMMON_HPP
