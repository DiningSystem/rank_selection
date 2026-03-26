#!/bin/bash

# ==========================================
# MoE-LoRA Merge + Evaluation Script
# ==========================================

MODEL="mistralai/Mistral-7B-v0.1"
GPU_ID=0

RUN_DIRS=()

if [ "$#" -gt 0 ]; then
    RUN_DIRS=("$@")
fi

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "Error: No run directories provided."
    echo "Usage:"
    echo "  bash scripts/arithmetic_merge_eval_moe.sh /path/to/run_dir [...]"
    exit 1
fi

for RAW_RUN_DIR in "${RUN_DIRS[@]}"; do

    RUN_DIR="$(printf '%s' "$RAW_RUN_DIR" | sed 's/\r$//')"
    RUN_DIR="${RUN_DIR%/}"

    echo "=========================================="
    echo "Processing: $RUN_DIR"
    echo "=========================================="

    METHOD=$(basename "$(dirname "$RUN_DIR")")
    echo "Method: $METHOD"

    # Handle both run_dir and direct final_model input
    if [ "$(basename "$RUN_DIR")" = "final_model" ]; then
        FINAL_MODEL_PATH="$RUN_DIR"
        RUN_ROOT="$(dirname "$RUN_DIR")"
    else
        FINAL_MODEL_PATH="$RUN_DIR/final_model"
        RUN_ROOT="$RUN_DIR"
    fi

    MERGED_MODEL_PATH="$RUN_ROOT/merged_model"

    echo "Final model path: $FINAL_MODEL_PATH"
    echo "Merged model path: $MERGED_MODEL_PATH"

    # ==========================
    # Merge step
    # ==========================
    echo "=== Starting MoE Merge ==="

    if [ ! -d "$FINAL_MODEL_PATH" ]; then
        echo "Error: final_model not found at $FINAL_MODEL_PATH"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python merge_moe_lora.py \
        --model_path "$FINAL_MODEL_PATH" \
        --base_model "$MODEL" \
        --output_path "$MERGED_MODEL_PATH"

    if [ ! -d "$MERGED_MODEL_PATH" ]; then
        echo "Error: Merge failed, no merged_model found"
        continue
    fi

    # ==========================
    # Evaluation
    # ==========================
    echo "=== Starting Evaluation ==="

    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/gsm8k_eval.py \
        --model "$MERGED_MODEL_PATH" \
        --tokenizer "$MERGED_MODEL_PATH" \
        --data_file "data/math_eval/gsm8k_test.jsonl" \
        --batch_size 128 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_ROOT"

    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/MATH_eval.py \
        --model "$MERGED_MODEL_PATH" \
        --tokenizer "$MERGED_MODEL_PATH" \
        --data_file "data/math_eval/MATH_test.jsonl" \
        --batch_size 64 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_ROOT"

    # ==========================
    # Cleanup
    # ==========================
    echo "=== Cleaning up merged model ==="

    if [ -d "$MERGED_MODEL_PATH" ]; then
        rm -rf "$MERGED_MODEL_PATH"
        if [ $? -eq 0 ]; then
            echo "Merged model removed successfully"
        else
            echo "Warning: failed to remove merged model"
        fi
    fi

    # ==========================
    # Logging
    # ==========================
    echo "Saving run info..."

    cat << EOF > "$RUN_ROOT/run_info.txt"
Run processed at: $(date)
Method: $METHOD
Base model: $MODEL
Merge: MoE-LoRA → dense
EOF

    echo "Done: $RUN_DIR"
done

echo "=========================================="
echo "All runs completed"
echo "=========================================="