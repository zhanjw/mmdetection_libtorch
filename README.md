## How to use

1. modified mmcv
comment
@`mmcv/cnn/bricks/conv_module.py`
```python
# @property
# def norm(self):
#     return getattr(self, self.norm_name)
```

2. modified mmdet
	- add function `forward_trace` to file `mmdet\models\detectors\single_stage.py`.
	See file `tools/single_stage.py` for implementation.
	- add function `forward_trace` to file `mmdet\models\detectors\two_stage.py`.
	See file `tools/two_stage.py` for implementation.

3. trace torchschript model using
	- `tools/get_trace_fasterrcnn.py`
	- `tools/get_trace_ssd_retinanet.py`
**Note:**
comment
@`mmdet/models/detectors/base.py`
```
# @property
# def with_shared_head(self):
#     """bool: whether the detector has a shared head in the RoI Head"""
#     return hasattr(self.roi_head,
#                    'shared_head') and self.roi_head.shared_head is not None
# @property
# def with_bbox(self):
#     """bool: whether the detector has a bbox head"""
#     return ((hasattr(self.roi_head, 'bbox_head')
#              and self.roi_head.bbox_head is not None)
#             or (hasattr(self, 'bbox_head') and self.bbox_head is not None))
# @property
# def with_mask(self):
#     """bool: whether the detector has a mask head"""
#     return ((hasattr(self.roi_head, 'mask_head')
#              and self.roi_head.mask_head is not None)
#             or (hasattr(self, 'mask_head') and self.mask_head is not None))
```
when trace single_stage.py model.

4. build cpp projects
```bash
	mkdir build
	cd build
	cmake ..
	make -j8
```

5. run ./test_detector

## Environment
Python 3.8.5
Libtorch 1.5.1
CUDA: 10.2
mmdetection: v2.3.0
mmcv: 1.1.2
TRTorch build from sources
