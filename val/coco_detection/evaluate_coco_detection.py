import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main(annotation_file, result_file, img_ids=None):
    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(result_file)

    # 若未提供 img_ids，则从结果文件中自动提取
    if img_ids is None:
        with open(result_file) as f:
            dt = json.load(f)
        img_ids = sorted({ann['image_id'] for ann in dt})

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="Path to COCO ground-truth JSON, e.g., instances_val.json")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Path to detection results JSON")
    parser.add_argument("--img-ids", type=int, nargs='*',
                        help="Optional: explicit list of image ids to evaluate")
    args = parser.parse_args()

    main(args.annotation_file, args.result_file, args.img_ids)