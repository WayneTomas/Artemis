import os
import json
import re
import argparse
from tqdm import tqdm
import math
import torch
from sentence_transformers import SentenceTransformer, util
from pycocotools.coco import COCO

# ------------------- Qwen2.5-VL smart resize -------------------
def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280):
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

def unresize_bbox(bbox, orig_height, orig_width):
    """
    将 smart resize 后的 [x1,y1,x2,y2] bbox 转回原图尺度，
    并输出 [x,y,w,h] (COCO 格式)。
    """
    new_h, new_w = smart_resize(orig_height, orig_width)
    scale_w = new_w / orig_width
    scale_h = new_h / orig_height

    x1, y1, x2, y2 = bbox
    x1_orig = x1 / scale_w
    y1_orig = y1 / scale_h
    x2_orig = x2 / scale_w
    y2_orig = y2 / scale_h

    w_orig = x2_orig - x1_orig
    h_orig = y2_orig - y1_orig
    return [x1_orig, y1_orig, w_orig, h_orig]

# ------------------- COCO 类别 & 同义词映射 -------------------
coco = COCO("val/coco_detection/anno/instances_val2017.json")
cats = coco.loadCats(coco.getCatIds())
coco_cat2id = {c["name"]: c["id"] for c in cats}
coco_classes = [c['name'] for c in cats]

synonym_mapping = {
    "sofa": "couch",
    "couch": "couch",
    "man": "person",
    "woman": "person",
    "people": "person",
    "human": "person",
    "tv monitor": "tv",
    "television": "tv",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "puppy": "dog",
    "kitten": "cat"
}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def map_to_coco_class(pred_label, model=None, coco_embeddings=None):
    """优先映射到 COCO 类别"""
    label = normalize_text(pred_label)
    if label in coco_cat2id:
        return label
    if label in synonym_mapping:
        return synonym_mapping[label]

    # fallback: semantic match
    if model is not None and coco_embeddings is not None:
        pred_emb = model.encode([label], convert_to_tensor=True)
        cos_scores = util.cos_sim(pred_emb, coco_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        return coco_classes[best_idx]

    return None

# ------------------- 提取框函数 -------------------
from typing import List, Dict, Any

def extract_answer(response: str) -> List[Dict[str, Any]]:
    if not isinstance(response, str):
        return []

    # <answer> 标签
    answer_block = _first_tag_block(response, "answer")
    if answer_block:
        boxes = _extract_boxes(answer_block)
        if boxes:
            return boxes

    # 全局兜底
    return _extract_boxes(response)

def _first_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}\s*>(.*?)</{tag}\s*>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_boxes(text: str) -> List[Dict[str, Any]]:
    # 去掉 <answer> 和 ```json ``` 包裹
    text = re.sub(r"</?answer>|```json|```", "", text).strip()

    boxes = []
    # 匹配包含 bbox_2d / bbox 2d / bbox，label 可选，point_2d 可选
    obj_pat = re.compile(
        r'\{[^}]*?(?:"label"\s*:\s*"([^"]*)"\s*,)?'           # 可选 label
        r'[^}]*?"(bbox_2d|bbox 2d|bbox)"\s*:\s*(\[[^\]]+\])'  # bbox
        r'(?:\s*,\s*"point_2d"\s*:\s*(\[[^\]]+\]))?'           # 可选 point_2d
        r'[^}]*\}',
        re.DOTALL
    )

    for match in obj_pat.findall(text):
        label, _, bbox_str, point_str = match
        try:
            bbox = json.loads(bbox_str)
            point = json.loads(point_str) if point_str else None
            if isinstance(bbox, list) and len(bbox) == 4:
                boxes.append({
                    "label": label if label else None,
                    "bbox_2d": bbox,
                    "point_2d": point
                })
        except Exception:
            continue

    return boxes

# ------------------- 主逻辑 -------------------
def main():
    parser = argparse.ArgumentParser(description="Convert MLLM predictions to COCO format")
    parser.add_argument("--input-file", type=str, required=True, help="Path to merged prediction file (JSONL)")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save COCO format result")
    parser.add_argument("--use-semantic-match", action="store_true", help="Enable semantic match fallback")
    args = parser.parse_args()

    model, coco_embeddings = None, None
    if args.use_semantic_match:
        print("Loading Sentence-BERT model for semantic similarity...")
        model = SentenceTransformer('ckpts/all-MiniLM-L6-v2')
        coco_embeddings = model.encode(coco_classes, convert_to_tensor=True)

    results = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing predictions"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except:
                raise ValueError('Failed to parse JSON line')

            image_path = item.get("image", "")
            image_id = int(os.path.splitext(os.path.basename(image_path))[0])

            raw_response = item.get("response", "")
            if isinstance(raw_response, list):
                raw_response = "".join(raw_response)

            objects = extract_answer(raw_response)
            for obj in objects:
                pred_label = obj.get("label", "")
                bbox = obj.get("bbox_2d", None)
                if not bbox or len(bbox) != 4:
                    continue

                mapped_class = map_to_coco_class(pred_label, model, coco_embeddings)
                if mapped_class is None:
                    continue

                # ---- 关键修改：反变换回原图 ----
                img_info = coco.loadImgs(image_id)[0]
                orig_w, orig_h = img_info['width'], img_info['height']
                coco_bbox = unresize_bbox(bbox, orig_h, orig_w)

                results.append({
                    "image_id": image_id,
                    "category_id": coco_cat2id[mapped_class],
                    "bbox": coco_bbox,
                    "score": obj.get("score", 1.0)
                })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"✅ COCO format results saved to {args.output_file}, total {len(results)} boxes.")

if __name__ == "__main__":
    main()
