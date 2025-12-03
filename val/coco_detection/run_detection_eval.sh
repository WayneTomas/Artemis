#!/bin/bash
DATASET="coco"
SPLIT="val"
model_path="your_model_path"
model_name="your_model_name"
answers_base="val/coco_detection/outputs/${DATASET}/${SPLIT}/${model_name}"

echo "Creating output directory: $answers_base"
mkdir -p "$answers_base"

GPULIST=(${CUDA_VISIBLE_DEVICES//,/ })
CHUNKS=${#GPULIST[@]}

echo "GPUs allocated: ${GPULIST[*]}"
echo "Number of chunks: $CHUNKS"

for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Starting inference on GPU ${GPULIST[$IDX]} with chunk index $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} \
    python -u val/coco_detection/infer_artemis_detection.py \
        --model-path "${model_path}" \
        --question-file "val/coco_detection/anno/coco_val.jsonl" \
        --image-folder "your_image_folder" \
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

# transfer to COCO result format
coco_result="${answers_base}/coco_results.json"
python val/coco_detection/convert_to_coco_result.py --input-file "$merged_file" --output-file "$coco_result"

# call COCO official evaluation script
python val/coco_detection/evaluate_coco_detection.py \
    --annotation-file val/coco_detection/anno/instances_val2017.json \
    --result-file "$coco_result"
