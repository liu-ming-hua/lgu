#!/bin/bash

# Define model list and dataset list
model_list=(
    "Llama-2-7b-chat"
    "Llama-2-13b-chat"
    "Qwen3-32B"
    "falcon-7b"
    "falcon-7b-instruct"
    "Mistral-7B-v0.1"
) 

dataset_list=(
    "trivia_qa" "squad" "bioasq" "nq" 
    "hotpot_qa") # 

# Fixed parameters (do not modify)
fixed_params=(
    "--entailment_model=deberta"
    "--temperature=1"
    "--no-analyze_run"
    "--num_samples=400"  
    "--metric=llm_gpt-3.5"
    "--num_generations=10"
)

# Nested loop to iterate all model-dataset combinations
for model in "${model_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do
        echo "========================================"
        echo "Start running: model=$model | dataset=$dataset"
        echo "========================================"
        
        # Execute core command
        python semantic_uncertainty/generate_answers.py \
            --model_name="$model" \
            --dataset="$dataset" \
            "${fixed_params[@]}"
        
        # Check if command executed successfully
        if [ $? -eq 0 ]; then
            echo "✅ Finished: model=$model | dataset=$dataset"
        else
            echo "❌ Failed: model=$model | dataset=$dataset"
            # Optional: To continue after failure, keep the comment below; to stop on failure, change to exit 1
            # exit 1
        fi
        
        echo -e "\n"  # Output blank line to separate runs
    done
done

echo "🎉 All combinations have finished running!"