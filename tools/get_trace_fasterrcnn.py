import argparse

from mmcv import Config

from mmdet.models import build_detector

import torch

import matplotlib.pyplot as plt
import mmcv
import warnings
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--checkpoint', help='the checkpoint file to trace')
    parser.add_argument(
        '--tracedbone', help='the name of tracedpoint')
    parser.add_argument(
        '--tracedshared', help='the name of tracedpoint')
    parser.add_argument(
        '--tracedbbox', help='the name of tracedpoint')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1333, 800],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()

    if hasattr(model, 'forward_trace'):
        model.forward = model.forward_trace
    else:
        raise NotImplementedError

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("=========================tracedbone===============================")
    img = torch.rand(input_shape).cuda()
    traced_bone = torch.jit.trace(model, img)
    traced_bone.save(args.tracedbone)

    bbox_feats = torch.rand(1000, 256, 7, 7).cuda()
    if model.with_shared_head:
        print("=========================shared_head===============================")
        traced_shared = torch.jit.trace(model.roi_head.shared_head, bbox_feats)
        traced_shared.save(args.tracedshared)

    print("==========================bbox_head================================")
    traced_bbox = torch.jit.trace(model.roi_head.bbox_head, bbox_feats)
    traced_bbox.save(args.tracedbbox)

    # print("=====================inference_detector===========================")
    # from mmdet.apis import inference_detector, init_detector
    # model = init_detector(
    #     args.config, args.checkpoint, device=torch.device('cuda', 0))
    # result = inference_detector(model, "/run/media/eric/DATA/industrial/mmdetection/demo/demo.jpg")
    # print(result[0].shape)
    # print(result[0])


if __name__ == '__main__':
    main()
