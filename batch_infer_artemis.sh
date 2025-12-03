#!/bin/bash

DATASET="your_dataset"
level="baseline"

model_path="your_model_path"
model_name="your_model_name"

answers_base="your_answer_path"

echo "Creating output directory: $answers_base"
mkdir -p "$answers_base"

# ------------------------------------------------------------------
# Automatically obtain the list of GPUs allocated to this job
# Example: "4,5" -> (4 5)
GPULIST=(${CUDA_VISIBLE_DEVICES//,/ })
CHUNKS=${#GPULIST[@]}   # Number of GPUs
# ------------------------------------------------------------------

echo "GPUs allocated: ${GPULIST[*]}"
echo "Number of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Launching inference on GPU ${GPULIST[$IDX]} with chunk index $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
    python -u infer_artemis.py \
        --model-path "${model_path}" \
        --prompt none \
        --question-file "your_question_file.jsonl" \
        --image-folder "your_image_folder" \
        --answers-file "${answers_base}/${IDX}_answer.jsonl" \
        --chunk-idx "$IDX" \
        --num-chunks "$CHUNKS" \
        --temperature 0.0 \
        --max_new_tokens 1024 &
done

wait
