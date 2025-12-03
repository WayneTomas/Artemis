"""
Description: This is an official release version of the evaluation code for the Artemis visual grounding task.
Module name: grd_eval_utils.py, version: 1.0.0
Function: different methods for handling grounding tasks and computing IoU and results from model responses

Authors: Vi-ocean - Wei Tang
Creation Date: Aug 1, 2025
Last Modified: Dec 2, 2025
Version: release - V1.0

Modification History:
- Dec 2, 2025 - Wei Tang - release version - V1.0
"""

from PIL import Image
import torch
from torchvision.ops.boxes import box_area
import os
import re
import torch.nn.functional as F
import numpy as np
import math

# Qwen2.5-VL smart resize
def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width)/min(height,width)}")
    
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

# Resize a single bbox
def resize_bbox(bbox, orig_height, orig_width):
    new_h, new_w = smart_resize(orig_height, orig_width)
    scale_w = new_w / orig_width
    scale_h = new_h / orig_height
    x1, y1, x2, y2 = bbox
    x1_new = max(0, min(round(x1 * scale_w), new_w - 1))
    y1_new = max(0, min(round(y1 * scale_h), new_h - 1))
    x2_new = max(0, min(round(x2 * scale_w), new_w - 1))
    y2_new = max(0, min(round(y2 * scale_h), new_h - 1))
    return [x1_new, y1_new, x2_new, y2_new]

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union



class GRDUtils:
    def __init__(self, image_folder, model_type, iou):
        self.image_folder = image_folder
        self.model_type = model_type
        self.iou = iou
        if self.model_type in ["qwen2_5"]:
            self.PATTERN = re.compile(r'\[(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\]')
        else:
            raise ValueError("Invalid model type")

    def eval(self, line):
        if self.model_type in ["qwen2_5"]:
            self.eval_qwen2_5(line)
        else:
            raise ValueError("Invalid model type")


    def eval_qwen2_5(self, line):
        try:
            if '<answer>' in line['response'][-1]:
                content_answer = line['response'][-1].split('<answer>', 1)[-1].strip()
                m = re.search(r'\[(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\]', content_answer)
                predict_bbox = list(map(float, m.groups())) if m else [0.0, 0.0, 0.0, 0.0]
            else:
                predict_bbox = [0,0,0,0]
            
            gt_box = resize_bbox(line['bbox'], orig_height=line['height'], orig_width=line['width'])
            target_bbox = torch.tensor(gt_box,
                                       dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, 
                                       dtype=torch.float32).view(-1, 4)
            img_path = line["image"]
            # image = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')

            iou, _ = box_iou(predict_bbox, target_bbox)
            
            iou = iou.item()
            if iou >= self.iou:
                line['correct'] = True
            else:
                line['correct'] = False
        except:
            line['correct'] = False
