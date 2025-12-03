"""
Description: This is an official release version of the evaluation code for the Artemis visual grounding task.
Module name: evaluate_release.py, version: 1.0.0
Function: different methods for handling grounding tasks and computing IoU and results from model responses

Authors: Vi-ocean - Wei Tang
Creation Date: Aug 1, 2025
Last Modified: Dec 2, 2025
Version: release - V1.0

Modification History:
- Dec 2, 2025 - Wei Tang - release version - V1.0
"""

from tqdm import tqdm
from utils import save_jsonl, load_jsonl
import argparse
from grd_eval_utils import GRDUtils

def evaluate(answer_file, args, regen_answer=False):
    lines = load_jsonl(answer_file)
    grd_utils = GRDUtils(args.image_folder, args.model_type, args.iou)

    for i, line in enumerate(tqdm(lines)):
        # if i == 10796:
        #     print()
        grd_utils.eval(line)
    save_jsonl(answer_file, lines, t_stamp=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    parser.add_argument("--model_type", type=str, default="qwen2_5")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--cot_rec_err_threhold", type=float, default=200)
    args = parser.parse_args()

    evaluate(args.answers_file, args)