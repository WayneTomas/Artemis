import json
import re
import argparse

def extract_answer_count(resp):
    """从 <answer> 中提取数字"""
    ans_count = None
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", resp, re.S)
    if match:
        try:
            ans_dict = json.loads(match.group(1))
            values = list(ans_dict.values())
            if values:
                ans_count = int(values[0])
        except Exception:
            ans_count = None
    return ans_count

def extract_think_count(resp):
    """统计 <think> 中对象数量，不解析具体格式"""
    match = re.search(r"<think>\s*(\[[\s\S]*?\])\s*</think>", resp, re.S)
    if not match:
        return None
    content = match.group(1).strip()
    if not content or content == "[]":
        return 0
    objects = re.findall(r"\{.*?\}", content, re.S)
    return len(objects)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    args = parser.parse_args()

    input_file = args.input_file

    # 读取 JSONL
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    total = len(data)
    think_correct_count = 0
    answer_correct_count = 0
    combined_correct_count = 0
    combined_fallback_count = 0  # 使用 think 兜底的数量

    for item in data:
        gt = item.get("ground_truth")
        resp = item.get("response", [""])[0]

        ans_count = extract_answer_count(resp)
        think_count = extract_think_count(resp)

        # Think ACC
        think_correct = think_count == gt
        if think_correct:
            think_correct_count += 1

        # Answer ACC
        answer_correct = ans_count == gt
        if answer_correct:
            answer_correct_count += 1

        # Combined ACC (Answer优先，Think兜底)
        correct = False
        used_think_as_fallback = False
        if answer_correct:
            correct = True
        elif think_correct:
            correct = True
            used_think_as_fallback = True

        if correct:
            combined_correct_count += 1
        if used_think_as_fallback:
            combined_fallback_count += 1

        # 更新原始数据
        item["think_correct"] = think_correct
        item["answer_correct"] = answer_correct
        item["correct_combined"] = correct
        item["used_think_as_fallback"] = used_think_as_fallback

    # 计算 ACC
    think_acc = think_correct_count / total if total > 0 else 0
    answer_acc = answer_correct_count / total if total > 0 else 0
    combined_acc = combined_correct_count / total if total > 0 else 0
    fallback_ratio = combined_fallback_count / total if total > 0 else 0

    print(f"Think ACC: {think_acc:.4f} ({think_correct_count}/{total})")
    print(f"Answer ACC: {answer_acc:.4f} ({answer_correct_count}/{total})")
    print(f"Combined ACC (Answer优先 + Think兜底): {combined_acc:.4f} ({combined_correct_count}/{total})")
    print(f"Used think as fallback: {combined_fallback_count}/{total} ({fallback_ratio:.4f})")

    # 覆盖原 JSONL 文件
    with open(input_file, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
