"""
Description: This is an official release version of the evaluation code for the Artemis visual grounding task.
Module name: utils.py, version: 1.0.0
Function: different methods for handling grounding tasks and computing IoU and results from model responses

Authors: Vi-ocean - Wei Tang
Creation Date: Aug 1, 2025
Last Modified: Dec 2, 2025
Version: release - V1.0

Modification History:
- Dec 2, 2025 - Wei Tang - release version - V1.0
"""
import os
import json
from tqdm import tqdm
# from latex2sympy2 import latex2sympy
import numpy as np
import re
from copy import deepcopy
import time
from math import *

import json
from tqdm import tqdm  # Assuming tqdm is imported to show progress bar
import time
def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def save_jsonl(path: str, data: list, t_stamp=True) -> None:
    if t_stamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def load_jsonl(path: str):
    data= []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            parsed = json.loads(line)
            if parsed is not None:
                data.append(parsed)
            else:
                # 可选：打印 / 记录失败行
                print(f"[WARN] Line {lineno} skipped due to invalid JSON.")
    return data
