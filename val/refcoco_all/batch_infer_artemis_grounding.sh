#!/bin/bash
DATASET="refcoco" # refcoco, refcoco+, refcocog
SPLIT="${DATASET}_val"
level="baseline"
model_path="your_model_path"
model_name="your_model_name"
answers_base="val/refcoco_all/outputs/${DATASET}/${SPLIT}_${level}/${model_name}"

echo "Creating output directory: $answers_base"
mkdir -p "$answers_base"

# ------------------------------------------------------------------
# Automatically obtain the list of GPUs allocated to this job
# Example: "4,5" -> (4 5)
GPULIST=(${CUDA_VISIBLE_DEVICES//,/ })
CHUNKS=${#GPULIST[@]}
# ------------------------------------------------------------------

echo "GPUs allocated: ${GPULIST[*]}"
echo "Number of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Starting inference on GPU ${GPULIST[$IDX]} with chunk index $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
    python -u val/refcoco_all/infer_artemis.py \
        --model-path "${model_path}" \
        --prompt none \
        --question-file "val/refcoco_all/${DATASET}/${SPLIT}.jsonl" \
        --image-folder "your_image_path" \
        --answers-file "${answers_base}/${IDX}_answer.jsonl" \
        --chunk-idx "$IDX" \
        --num-chunks "$CHUNKS" \
        --temperature 0.0 \
        --max_new_tokens 1024 &
done
wait

# 合并所有分片结果
output_file="${answers_base}/merge.jsonl"
echo "Merging results into: $output_file"
> "$output_file"  # 清空或创建目标文件
cat ${answers_base}/*_answer.jsonl >> "$output_file"

echo "Merged output at $output_file"

# 评估时指定模型名方便结果管理
echo "Starting evaluation with answers file: $output_file"
python val/refcoco_all/evaluate_release.py --image_folder your_image_path --answers_file "$output_file" --model_type qwen