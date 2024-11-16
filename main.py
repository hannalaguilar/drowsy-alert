import mmcv
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
import matplotlib.pyplot as plt

det_config = 'retinanet_r50_fpn_1x_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

det_model = init_detector(det_config, det_checkpoint, device='cpu')
