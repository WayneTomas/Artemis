import json
import os

input_file = "val/coco_detection/anno/instances_val2017.json"  # COCO官方标注
output_file = "val/anno/coco_detection/coco_val.jsonl"
image_prefix = "coco/train2017"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, "r") as f:
    data = json.load(f)

images = {img["id"]: img["file_name"] for img in data["images"]}

with open(output_file, "w") as f_out:
    for img_id, img_name in images.items():
        item = {
            "image": f"{image_prefix}/{img_name}",
            "sent": "List all visible objects in the image with their bounding boxes."  # prompt固定
        }
        f_out.write(json.dumps(item) + "\n")

print(f"Saved {len(images)} entries to {output_file}")
