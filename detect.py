import os
import sys


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.inferer import Inferer

weights='yolov6n.pt'
source='data/images'
yaml="data/coco.yaml"
img_size=640
conf_thres=0.25
iou_thres=0.45
device='cpu'
save_img=True
classes=None
agnostic_nms=False
max_det=1000
save_dir=False 
save_txt=False 
save_img=True
hide_labels=False
hide_conf=False
half=False

# create save dir
save_dir = "run"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Inference
inferer = Inferer(source, weights, device, yaml, img_size, half)
inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf)


inferer = Inferer(source, weights, device, yaml, img_size, False)
inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf)
