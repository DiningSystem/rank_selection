#!/bin/bash

# Configuration
MODEL="mistralai/Mistral-7B-v0.1"
GPU_ID=0  # Specify which GPU to use

# List of run directories to process; add trained adapter directories here.
# You can also pass run dirs as CLI args:
#   bash scripts/arithmetic_merge_eval.sh /path/to/run_dir [/path/to/run_dir2 ...]
RUN_DIRS=()

if [ "$#" -gt 0 ]; then
    RUN_DIRS=("$@")
fi

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "Error: No run directories provided."
    echo "Provide one or more run dirs as args, e.g.:"
    echo "  bash scripts/arithmetic_merge_eval.sh /abs/path/to/run_dir"
    exit 1
fi

# Process each run directory
for RAW_RUN_DIR in "${RUN_DIRS[@]}"; do
    # Normalize accidental CRLF/trailing slash from pasted paths.
    RUN_DIR="$(printf '%s' "$RAW_RUN_DIR" | sed 's/\r$//')"
    RUN_DIR="${RUN_DIR%/}"
    echo "=== Processing: $RUN_DIR ==="
    
    # Extract method from the directory structure (one folder above run directory)
    METHOD=$(basename "$(dirname "$RUN_DIR")")
    echo "Extracted method: $METHOD"
    
    # Get the final and merged model paths.
    # Accept both:
    #   - run directory (.../timestamp_rank_...): expects /final_model under it
    #   - final_model directory directly (.../final_model)
    if [ "$(basename "$RUN_DIR")" = "final_model" ]; then
        FINAL_MODEL_PATH="$RUN_DIR"
    else
        FINAL_MODEL_PATH="$RUN_DIR/final_model"
    fi
    MERGED_MODEL_PATH="$RUN_DIR/merged_model"
    
    echo "=== Starting Merging ==="
    if [ ! -d "$FINAL_MODEL_PATH" ]; then
        echo "Error: Final model not found at $FINAL_MODEL_PATH"
        echo "Debug: showing parent directory listing"
        ls -la "$(dirname "$FINAL_MODEL_PATH")" 2>/dev/null || true
        continue
    fi
        
    CUDA_VISIBLE_DEVICES=$GPU_ID python merge_save.py \
        --base_model "$MODEL" \
        --adapter_path "$FINAL_MODEL_PATH" \
        --output_path "$MERGED_MODEL_PATH"
    
    echo "=== Starting Evaluation ==="
    # Check if merged model exists
    if [ ! -d "$MERGED_MODEL_PATH" ]; then
        echo "Error: Merged model not found at $MERGED_MODEL_PATH"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/gsm8k_eval.py \
        --model "$MERGED_MODEL_PATH" \
        --data_file "data/math_eval/gsm8k_test.jsonl" \
        --batch_size 128 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_DIR" \

    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/MATH_eval.py \
        --model "$MERGED_MODEL_PATH" \
        --data_file "data/math_eval/MATH_test.jsonl" \
        --batch_size 64 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_DIR" \

    # Clean up merged model directory
    echo "=== Cleaning up merged model ==="
    if [ -d "$MERGED_MODEL_PATH" ]; then
        echo "Removing merged model directory: $MERGED_MODEL_PATH"
        rm -rf "$MERGED_MODEL_PATH"
        if [ $? -eq 0 ]; then
            echo "Successfully removed merged model directory"
        else
            echo "Warning: Failed to remove merged model directory"
        fi
    else
        echo "Merged model directory not found - nothing to clean up"
    fi
    
    echo "=== Processing Complete for $RUN_DIR ==="
    
    # Save run information
    echo "Saving run information..."
    cat << EOF > "$RUN_DIR/run_info.txt"
Run processed at: $(date)
Method: $METHOD
Base model: $MODEL
EOF

done

echo "=== All Run Directories Processed ==="
