#!/bin/bash

batch_sizes=(8 16 32 64)
max_tokens_nums=(64 128 256 512)
bert_pretrained_models=(prajjwal1/bert-small prajjwal1/bert-medium bert-base-uncased)
bert_pretrained_models_names=(bert-small bert-medium bert-base-uncased)

for model_index in ${!bert_pretrained_models[*]}
do
    model=${bert_pretrained_models[$model_index]}
    model_name=${bert_pretrained_models_names[$model_index]}

    for batch_size in "${batch_sizes[@]}"
    do
        for max_tokens in "${max_tokens_nums[@]}"
        do
            log_file_path="./bert_logs/max_tokens=${max_tokens}_batch_size=${batch_size}_pretrained_model=${model_name}.log"
            
            python3 bert_approach.py --epochs 5 --batch_size ${batch_size} --pretrained_model ${model} --max_tokens ${max_tokens} \
                                     --log ${log_file_path}
            
            echo ""
            echo "${log_file_path} results"
            grep -iP "(Accuracy|F1) on (Train|Test) Set" ${log_file_path}
        done
    done
done