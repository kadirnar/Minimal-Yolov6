import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'YOLOv6') not in sys.path:
    sys.path.append(str(ROOT / 'YOLOv6'))  # add YOLOv6 ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from YOLOv6.yolov6.core.inferer import Inferer
import os.path as osp
import glob
from tqdm import tqdm

import numpy as np
import cv2
import torch

from YOLOv6.yolov6.utils.events import load_yaml
from YOLOv6.yolov6.layers.common import DetectBackend
from YOLOv6.yolov6.utils.nms import non_max_suppression
from YOLOv6.yolov6.data.datasets import IMG_FORMATS
from YOLOv6.yolov6.utils.events import LOGGER

class Yolov6:
    def __init__(
        self,
        model_path: str,
        device: str,
        img_size: int = 640,
        source : str = 'YOLOv6/data/images/image1.jpg',
        conf_thres: float = 0.3,
        iou_thres: float = 0.5,
        yaml : str = 'YOLOv6/data/coco.yaml',
        
    ):  
        self.half = False
        self.model_path = model_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.yaml = yaml
        self.source = source
        self.load_model()

    def load_model(self):
        self.model = DetectBackend(self.model_path, self.device)
        self.stride = self.model.stride
        
        # Half precision
        if self.half:
            self.model.model.half()
        else:
            self.model.model.float()
            half = False
        
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup


        
    def inference(self, classes=None, agnostic_nms=False, max_det=1000, save_dir=None, save_txt=False, save_img=True, hide_labels=False, hide_conf=False):
        # create save dir
        project, name = osp.join(ROOT, 'runs/inference'), "exp"
        save_dir = osp.join(project, name)

        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        else:
            LOGGER.warning('Save directory already existed')    
            
        self.yaml = load_yaml(self.yaml)['names']    
        
        # Load data
        if os.path.isdir(self.source):
            img_paths = sorted(glob.glob(os.path.join(self.source, '*.*')))  # dir
        elif os.path.isfile(self.source):
            img_paths = [self.source]  # files
        else:
            raise Exception(f'Invalid path: {self.source}')
        img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1].lower() in IMG_FORMATS]
    
        for img_path in tqdm(img_paths):
            img, img_src = Inferer.precess_image(img_path, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            pred = self.model(img)
            det = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            
            save_path = osp.join(save_dir, osp.basename(img_path))  # im.jpg
            txt_path = osp.join(save_dir, 'labels', osp.basename(img_path).split('.')[0])
            
            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src


            if len(det):
                det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (Inferer.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:
                        class_num = int(cls)  # integer class
                        label = None if hide_labels else (self.yaml[class_num] if hide_conf else f'{self.yaml[class_num]} {conf:.2f}')

                        Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.001), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))

                img_src = np.asarray(img_ori)

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, img_src)
                
        
detection_model = Yolov6(
    model_path="yolov6s.pt",
    source="1.jpg",
    img_size=1280,
    conf_thres=0.3,
    device="cpu",
)

detection_model.inference()
