//
// Created by dl on 2020/4/15.
//

#include "bbox.hpp"
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")


at::Tensor nms(const at::Tensor& dets, const float threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda(dets, threshold);
}

torch::Tensor delta2bbox(const torch::Tensor& rois, const torch::Tensor& deltas,
                         const std::vector<int>& max_shape,
                         const std::vector<float>& means, const std::vector<float>& stds,
                         float wh_ratio_clip) {

    torch::Tensor Tmeans = torch::tensor(means).to(deltas.device());
    Tmeans = Tmeans.repeat({1, deltas.sizes()[1] / 4});
    torch::Tensor Tstds = torch::tensor(stds).to(deltas.device());
    Tstds = Tstds.repeat({1, deltas.sizes()[1] / 4});

    double max_ratio = abs(log(wh_ratio_clip));
    torch::Tensor denorm_deltas = deltas * Tstds + Tmeans;

    torch::Tensor dx = denorm_deltas.slice(1, 0, deltas.sizes()[1], 4);
    torch::Tensor dy = denorm_deltas.slice(1, 1, deltas.sizes()[1], 4);
    torch::Tensor dw = denorm_deltas.slice(1, 2, deltas.sizes()[1], 4).clamp(-max_ratio, max_ratio);
    torch::Tensor dh = denorm_deltas.slice(1, 3, deltas.sizes()[1], 4).clamp(-max_ratio, max_ratio);

    torch::Tensor px = ((rois.select(1, 0) + rois.select(1, 2)) * 0.5).unsqueeze(1).expand_as(dx);
    torch::Tensor py = ((rois.select(1, 1) + rois.select(1, 3)) * 0.5).unsqueeze(1).expand_as(dy);
    torch::Tensor pw = (rois.select(1, 2) - rois.select(1, 0)).unsqueeze(1).expand_as(dw);
    torch::Tensor ph = (rois.select(1, 3) - rois.select(1, 1)).unsqueeze(1).expand_as(dh);

    torch::Tensor gw = pw * dw.exp();
    torch::Tensor gh = ph * dh.exp();

    torch::Tensor gx = px + 1 * pw * dx;
    torch::Tensor gy = py + 1 * ph * dy;

    torch::Tensor x1 = (gx - gw * 0.5).clamp(0.0, max_shape[1]);
    torch::Tensor y1 = (gy - gh * 0.5).clamp(0.0, max_shape[0]);
    torch::Tensor x2 = (gx + gw * 0.5).clamp(0.0, max_shape[1]);
    torch::Tensor y2 = (gy + gh * 0.5).clamp(0.0, max_shape[0]);
    return torch::stack({x1, y1, x2, y2}, -1).view_as(deltas);
}

torch::Tensor multiclass_nms(const torch::Tensor& multi_bboxes,
                             const torch::Tensor& multi_scores,
                             float score_thr, float iou_thr,
                             int max_num) {
    torch::Tensor bboxes = torch::zeros({0, 5}).cuda();
    torch::Tensor labels = torch::zeros({0, 1}).cuda();
    int num_classes = multi_scores.size(1) - 1;
    torch::Tensor multi_bboxes_reshape;
    if (multi_bboxes.size(1) > 4) {
        multi_bboxes_reshape = multi_bboxes.view({multi_scores.size(0), -1, 4});
    } else {
        multi_bboxes_reshape = multi_bboxes.unsqueeze(1).expand({multi_scores.size(0), num_classes, 4});
    }
    torch::Tensor scores = multi_scores.slice(1,0,4).reshape({-1,1});
    for(int i = 0; i < num_classes; i++) {
        torch::Tensor cls_inds = multi_scores.select(1, i) > score_thr;
        if (torch::any(cls_inds).item().toBool() == 0) {
            continue;
        }
        cls_inds = torch::nonzero(cls_inds).squeeze();
        torch::Tensor _bboxes = multi_bboxes_reshape.select(1, i).index_select(0, cls_inds);
//        torch::Tensor _bboxes = multi_bboxes.index_select(0, cls_inds);
        torch::Tensor _scores = multi_scores.index_select(0, cls_inds).select(1, i).unsqueeze(1);
        torch::Tensor cls_dets = torch::cat({_bboxes, _scores}, 1);
        torch::Tensor inds = nms(cls_dets, iou_thr);
        cls_dets = cls_dets.index_select(0, inds);
        torch::Tensor cls_labels = multi_bboxes_reshape.new_full({cls_dets.sizes()[0], 1}, i);
//        torch::Tensor cls_labels = multi_bboxes.new_full({cls_dets.sizes()[0], 1}, i);
            bboxes = torch::cat({bboxes, cls_dets}, 0);
            labels = torch::cat({labels, cls_labels}, 0);
    }
        if (max_num > 0 && bboxes.sizes()[0] > max_num) {
            bboxes = bboxes.slice(0, 0 , max_num);
            labels = labels.slice(0, 0, max_num);
        }

    return torch::cat({bboxes, labels}, 1);
}

torch::Tensor singleclass_nms(const torch::Tensor& proposals,float iou_thr) {

    torch::Tensor inds = nms(proposals, iou_thr);
    return inds;
}

void bbox2result(torch::Tensor& result, float thresh, cv::Size2f scale,
                 std::vector<DetectedBox>& detected_boxes) {

    if (result.sizes()[0] == 0) {
        return;
    }
    result = result.cpu();
    // Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar type and
    auto result_data = result.accessor<float, 2>();
    for (int64_t i = 0; i < result.size(0) ; i++)
    {
        float score = result_data[i][4];
        if (score > thresh) {
            DetectedBox detected_box;
            detected_box.box.x = result_data[i][0] * scale.width ;
            detected_box.box.y = result_data[i][1] * scale.height  ;
            detected_box.box.width = (result_data[i][2] - result_data[i][0]) * scale.width;
            detected_box.box.height = (result_data[i][3] - result_data[i][1]) * scale.height;
            detected_box.label = result_data[i][5];
            detected_box.score = score;
            detected_boxes.emplace_back(detected_box);
        }
    }
}


//only one image
torch::Tensor bbox2roi(const torch::Tensor& proposals) {
    torch::Tensor img_inds = proposals.new_full({proposals.sizes()[0],1}, 0);
    torch::Tensor rois = torch::cat({img_inds, proposals.slice(1, 0, 4)}, 1);
    return rois;
}










