#!/bin/bash

# Define model list and dataset list
model_list=(
    "Llama-2-7b-chat"
    "Qwen3-32B"
    "Llama-2-13b-chat"
    "falcon-7b"
    "falcon-7b-instruct"
    "Mistral-7B-v0.1"
) 

dataset_list=(
    "trivia_qa" "squad" "bioasq" "nq"
) 

for model in "${model_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do
        # Concatenate model name and dataset name as run_id
        runid="${model}_${dataset}"
        
        echo "========================================"
        echo "Start running: model=$model | dataset=$dataset | run_id=$runid"
        echo "========================================"
    
        # Execute core command
        python /root/lmh/LGU/semantic_uncertainty/compute_lgu.py \
            --run_id="$runid"
        
        # Check command execution result
        if [ $? -eq 0 ]; then
            echo "✅ Success: runid=$runid"
        else
            echo "❌ Failed: runid=$runid"
            # Optional: Stop script after failure? Keep the comment below to continue, uncomment to stop
            # exit 1
        fi
        
        echo -e "\n"  # Blank line to separate logs for different runids
    done
done

echo "🎉 All runids have been processed!"
