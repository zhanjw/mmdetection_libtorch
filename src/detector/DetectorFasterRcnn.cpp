#include "DetectorFasterRcnn.hpp"
#include "trtorch/trtorch.h"

void split(const std::string& s, std::vector<std::string>& tokens, char delim = ' ') {
    tokens.clear();
    auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < s.size(); i++) {
            if (s[i] != delim) return i;
        }
        return std::string::npos;
    };
    size_t lastPos = string_find_first_not(0);
    size_t pos = s.find(delim, lastPos);
    while (lastPos != std::string::npos) {
        tokens.emplace_back(s.substr(lastPos, pos - lastPos));
        lastPos = string_find_first_not(pos);
        pos = s.find(delim, lastPos);
    }
}


DetectorFasterRcnn::DetectorFasterRcnn(){

}

DetectorFasterRcnn::~DetectorFasterRcnn(){

}

void DetectorFasterRcnn::LoadParams(const Params& params, torch::DeviceType* device_type){
    DetectorCommon::LoadParams(params, device_type);
    single_roi_extractor_.init_params(roi_extractor_params_);
    std::vector<std::string> tokens;
    std::vector<std::string> save_tokens;
    split(model_path_, tokens, ',');
    split(save_path_, save_tokens, ',');
    assert (tokens.size() > 0);
    bone_model_path_ = exists(save_tokens[0])?save_tokens[0]:tokens[0];
    save_bone_model_path_ = save_tokens[0];
    if (tokens.size() == 2) {
        bbox_model_path_ = exists(save_tokens[1])?save_tokens[1]:tokens[1];
        save_bbox_model_path_ = save_tokens[1];
        with_shared_ = false;
    }
    if (tokens.size() == 3) {
       shared_model_path_ = exists(save_tokens[1])?save_tokens[1]:tokens[1];
        save_bbox_model_path_ = save_tokens[1];
       bbox_model_path_ = exists(save_tokens[2])?save_tokens[2]:tokens[2];
        save_bbox_model_path_ = save_tokens[2];
       with_shared_ = true;
    }

    get_anchor_generators({}, rpn_head_params_.anchor_scales_, rpn_head_params_.anchor_ratios_);
}


void DetectorFasterRcnn::LoadTracedModule() {
    bone_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(bone_model_path_));
    if (with_shared_) {
        shared_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(shared_model_path_));
    }
    bbox_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(bbox_model_path_));
}


void DetectorFasterRcnn::get_rpn_fpn_data(const torch::Tensor& output, torch::Tensor& rpn_data, std::vector<torch::Tensor>& fpn_datas) {
    int rpn_num = 0;
    int fpn_num = 0;
    int rpn_channels = 4 + rpn_head_params_.class_num_ -1;
    int fpn_channels = fpn_params_.out_channels_;
    for(int k = 0; k < int(anchor_generators_.size()); k++) {
        rpn_num += anchor_generators_[k].anchor_nums_ * rpn_channels;
        assert (anchor_generators_[k].feature_maps_sizes_.size() == 2);
        fpn_num += anchor_generators_[k].feature_maps_sizes_[0] * anchor_generators_[k].feature_maps_sizes_[1] * fpn_channels;
    }
    rpn_data = output.slice(0, 0, rpn_num).view({-1, rpn_channels});
/*
[217413, 5]
*/
    torch::Tensor fpn_data = output.slice(0, rpn_num, output.sizes()[0]);
    int start = 0;
    int end = 0;
    for(int k = 0; k < int(anchor_generators_.size()); k++) {
      int feature_height = anchor_generators_[k].feature_maps_sizes_[0];
      int feature_width = anchor_generators_[k].feature_maps_sizes_[1];
      start = end;
      int fpn_num_layer = feature_height * feature_width * fpn_channels;
      end = end + fpn_num_layer;
      torch::Tensor fpn_data_layer = fpn_data.slice(0, start, end).view({1, fpn_channels, feature_height, feature_width});
/*
[1, 256, 200, 272]
[1, 256, 100, 136]
[1, 256, 50, 68]
[1, 256, 25, 34]
[1, 256, 13, 17]
*/
      fpn_datas.push_back(fpn_data_layer);
    }
}


void DetectorFasterRcnn::Detect(const cv::Mat& image, std::vector<DetectedBox>& detected_boxes) {
    torch::Tensor tensor_image;
    transform(image, tensor_image, transform_params_, net_width_, net_height_, device_);

    //start trtorch
    if (!bone_module_->is_training()){
        *bone_module_ = trtorch::CompileGraph(*bone_module_, std::vector<trtorch::ExtraInfo::InputRange>({tensor_image.sizes()}));
        std::cout<<"save to: "<<save_bone_model_path_<<std::endl;
        bone_module_->save(save_bone_model_path_);
    }
    //end trtorch

    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = bone_module_->forward({tensor_image}).toTensor().squeeze(0);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    get_anchor_boxes();

    torch::Tensor rpn_data;
    std::vector<torch::Tensor> fpn_datas;
    get_rpn_fpn_data(output, rpn_data, fpn_datas);

    torch::Tensor proposals_scores;
    torch::Tensor proposals_bboxes;
    bool rpn = true;
    get_proposals(rpn_data, transform_params_.img_shape_, proposals_bboxes, proposals_scores,  rpn);

    torch::Tensor proposals = torch::cat({proposals_bboxes, proposals_scores}, 1);
    if (rpn_head_params_.nms_across_levels_ == 1) {
        torch::Tensor inds = singleclass_nms(proposals, anchor_head_params_.nms_thresh_);
        int num = cv::min(int(inds.sizes()[0]), rpn_head_params_.max_num_);
        proposals = proposals.slice(0, 0, num);
    } else {
        int num = cv::min(rpn_head_params_.max_num_, int(proposals_scores.sizes()[0]));
        std::tuple<torch::Tensor, torch::Tensor> topk = torch::topk(proposals_scores.squeeze(), num);
        torch::Tensor topk_inds = std::get<1>(topk);
        proposals = proposals.index_select(0, topk_inds);
    }

    torch::Tensor rois = bbox2roi(proposals);

    torch::Tensor roi_feats = single_roi_extractor_.bbox_roi_extractor(fpn_datas, rois);

    //start trtorch
    if (!bbox_module_->is_training()){
        *bbox_module_ = trtorch::CompileGraph(*bbox_module_, std::vector<trtorch::ExtraInfo::InputRange>({roi_feats.sizes()}));
        std::cout<<"save to: "<<save_bbox_model_path_<<std::endl;
        bbox_module_->save(save_bbox_model_path_);
    }
    //end trtorch

    auto bbox_head_output = bbox_module_->forward({roi_feats}).toTuple();

    torch::Tensor cls_score = bbox_head_output->elements()[0].toTensor();
    torch::Tensor bbox_pred = bbox_head_output->elements()[1].toTensor();

    std::vector<int> max_shape = {image.rows, image.cols};

    torch::Tensor scores = torch::softmax(cls_score, 1);

    torch::Tensor bboxes = delta2bbox(rois.slice(1, 1, 5), bbox_pred,
                                      transform_params_.img_shape_,
                                      anchor_head_params_.bbox_target_means_,
                                      anchor_head_params_.bbox_target_stds_);

    bboxes = bboxes.view({bboxes.size(0),-1,4});
    bboxes /= torch::tensor(transform_params_.vector_scale_factor_).cuda();
    torch::Tensor nms_results = multiclass_nms(bboxes, scores, anchor_head_params_.score_thresh_,
                                               0.5,anchor_head_params_.max_per_img_);

    cv::Size2f scale = cv::Size2f(1,1);
    bbox2result(nms_results, conf_thresh_, scale, detected_boxes);
}
