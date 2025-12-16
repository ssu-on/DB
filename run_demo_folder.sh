#!/bin/bash

IMAGE_DIR="assets/test4"
OUT_DIR="demo_results/batch_output4"
CFG="experiments/seg_detector/subtitles_resnet18_finetune.yaml"
WEIGHTS="outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet18/SubtitleBranchLoss-200/model/final"

mkdir -p "$OUT_DIR"

for img in "$IMAGE_DIR"/*.png; do
    if [[ -f "$img" ]]; then
        echo "Running inference on: $img"

        CUDA_VISIBLE_DEVICES=1 python demo.py \
            $CFG \
            --resume $WEIGHTS \
            --polygon \
            --box_thresh 0.7 \
            --visualize \
            --image_short_side 512 \
            --image_path "$img" \
            --result_dir "$OUT_DIR" \
            --dest binary \
            --save_binary_mask \
            --mask_source boxes
    fi
done
