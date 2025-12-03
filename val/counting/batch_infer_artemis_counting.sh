#!/bin/bash
DATASET="pixmo_count"
SPLIT="val"
model_path="your_model_path"
model_name="your_model_name"
answers_base="val/counting/outputs/${DATASET}/${SPLIT}/${model_name}"

echo "Creating output directory: $answers_base"
mkdir -p "$answers_base"

GPULIST=(${CUDA_VISIBLE_DEVICES//,/ })
CHUNKS=${#GPULIST[@]}

echo "GPUs allocated: ${GPULIST[*]}"
echo "Number of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Starting inference on GPU ${GPULIST[$IDX]} with chunk index $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
    python -u val/counting/infer_artemis_couning.py \
        --model-path "${model_path}" \
        --question-file "val/counting/jsons/counting/pixmo_count_${SPLIT}540.jsonl" \
        --image-folder "val/counting/images" \
        --answers-file "${answers_base}/${IDX}_answer.jsonl" \
        --chunk-idx "$IDX" \
        --num-chunks "$CHUNKS" \
        --temperature 0.0 \
        --max_new_tokens 1024 &
done
wait

merged_file="${answers_base}/merge.jsonl"
echo "Merging results into: $merged_file"
> "$merged_file"
cat ${answers_base}/*_answer.jsonl >> "$merged_file"

python val/counting/evaluate_counting.py --input_file "$merged_file"
