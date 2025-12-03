import argparse
import os
import json
import math
from tqdm import tqdm
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 选择一个固定模板组
bs = 32  # batch size


# Default perception prompts used in the Artemis Paper (https://arxiv.org/pdf/2512.01988)
# for grounding, detection, and counting tasks.
# The lisa grounding prompt is identical to the grounding prompt.
# For other tasks, please follow the original benchmark prompts or choose an appropriate one by your self.
def build_prompt(description: str, task_type: str) -> str:
    # 安全转义 description，避免引号、换行等问题
    safe_desc = json.dumps(description)[1:-1]  # 去掉首尾引号

    if task_type == "grounding":
        return f'''Analyze the image and answer the following question: "What region is described as: '{safe_desc}'?"

Think step by step:
1. Identify a small set of key objects or regions that help locate the described target.
2. These may include the target object itself and/or contextual objects (e.g., nearby items, reference persons, background cues).
3. Estimate their bounding box(es) in [x1, y1, x2, y2] format (pixel coordinates).

Output your reasoning in <think> tags as a JSON list of key objects (can include the target if useful):
<think>
[{{"label": "<object_label_1>", "bbox": [x1, y1, x2, y2]}},
 {{"label": "<object_label_2>", "bbox": [x1, y1, x2, y2]}}]
</think>

Then output the final answer in <answer> tags as a single dictionary for the target object:
<answer>
[{{"label": "{safe_desc}", "bbox": [x1, y1, x2, y2]}}]
</answer>'''

    elif task_type == "detection":
        categories = [
            "bird","kite","parking meter","oven","handbag","bed","fire hydrant","mouse","umbrella","truck",
            "knife","backpack","frisbee","giraffe","bottle","microwave","sheep","banana","car","traffic light",
            "dining table","person","baseball bat","bowl","spoon","boat","cake","baseball glove","bus","bench",
            "dog","cow","surfboard","snowboard","remote","cell phone","hair drier","book","suitcase","refrigerator",
            "potted plant","stop sign","clock","scissors","carrot","sandwich","skateboard","toothbrush","bicycle",
            "skis","toaster","laptop","sports ball","broccoli","fork","keyboard","train","pizza","teddy bear",
            "airplane","tennis racket","apple","orange","hot dog","bear","horse","cup","zebra","toilet","elephant",
            "wine glass","sink","motorcycle","donut","chair","tv","couch","tie","cat","vase"
        ]
        category_str = ", ".join(categories)
        return f'''Analyze the image and answer: "{safe_desc}"

Think step by step:
1. Identify all major visible objects in the image from the following category set: [{category_str}].
2. For each object, estimate its category and bounding box [x1, y1, x2, y2] (pixel coordinates).
3. Avoid duplicates and irrelevant objects.
4. Directly output the final answer in <answer> tags as a JSON list of all detected objects, and do not ouput any <think>:
<answer>
[{{"label": "<object_label_1>", "bbox": [x1, y1, x2, y2]}},
 {{"label": "<object_label_2>", "bbox": [x1, y1, x2, y2]}}]
</answer>'''
    elif task_type == "counting":
        return f'''Analyze the image and answer: "How many '{safe_desc}' are in the image?"

Step 1: Examine the image carefully and list **all visible instances** of "{safe_desc}" in <think> tags.
Step 2: Each instance should be a dictionary with "label" and "bbox" if possible.
Step 3: After listing all instances, count them.
Step 4: Output the total count in <answer> tags. **The number must exactly equal the number of items in <think>. Do NOT estimate or guess.**

Example:

<think>
[{{"label": "{safe_desc} #1", "bbox": [x1, y1, x2, y2]}},
 {{ "label": "{safe_desc} #2", "bbox": [x1, y1, x2, y2]}}]
</think>

<answer>
{{"{safe_desc} count": <number_of_items_in_think>}}
</answer>'''
    elif task_type == "no_think_counting":
        return f'''Analyze the image and answer: "How many '{safe_desc}' are in the image?"

Directly output the final answer in JSON format, and do not ouput any <think>:
Output the total count in <answer> tags.

Example:

<answer>
{{"{safe_desc} count": <number_of_items_in_think>}}
</answer>
'''

    else:
        raise ValueError(f"Unknown task_type: {task_type}. Supported: 'grounding', 'detection', 'counting'")


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    return split_list(lst, n)[k]

def batch(iterable, n=8):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def process_questions(questions, answers_file, num_chunks, chunk_idx):
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answers_file = os.path.join(os.path.dirname(answers_file), f"{chunk_idx}_{os.path.basename(answers_file)}")
    
    existing_answers = []
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                existing_answers.append(json.loads(line.strip()))
    
    existing_keys = {(q["image"], q["sent"], tuple(q.get("bbox", []))) for q in existing_answers}
    new_questions = [q for q in questions if (q["image"], q["sent"], tuple(q.get("bbox", []))) not in existing_keys]
    ans_file = open(answers_file, "a")
    return new_questions, ans_file

def main(args):
    assert torch.cuda.is_bf16_supported(), "GPU does not support bf16"
    dtype = torch.bfloat16

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    min_pixels = 56 * 56
    max_pixels = 14 * 14 * 4 * 1280
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = [json.loads(q) for q in f]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    questions, ans_file = process_questions(questions, answers_file, args.num_chunks, args.chunk_idx)

    for mini_batch in tqdm(list(batch(questions, bs)), total=math.ceil(len(questions)/bs)):
        messages_list = []
        for line in mini_batch:
            # 我们论文的最终prompt
            query = build_prompt(line["sent"], "detection")

            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            messages_list.append(messages)

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        for line, output in zip(mini_batch, outputs):
            line["response"] = [output]  # 保持 list 形式
            ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")
            ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, choices=['none', 'single', 'multimath'], default="none")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    main(args)
