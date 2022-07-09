#!/bin/bash

embedding_sizes=(50 150 300 500 1000)
hidden_state_sizes=(64 128 256 512 1024)
max_tokens_nums=(125 250 500 1000 2000)

for emb_size in "${embedding_sizes[@]}"
do
    for hid_state_size in "${hidden_state_sizes[@]}"
    do
        for max_tokens in "${max_tokens_nums[@]}"
        do
            log_file_path="./lstm_logs/max_tokens=${max_tokens}_hidden_state_size=${hid_state_size}_embedding_size=${emb_size}_no_stem_sr.log"
            
            # I will run the experiments without stemming and stopword removal
            # as I tried it and showed me that without doing those I have the
            # best performance

            python3 lstm_approach.py --embedding_size ${emb_size} --hidden_state_size ${hid_state_size} \
                                     --max_tokens ${max_tokens} --log ${log_file_path} --remove_stopwords
            
            echo ""
            echo "${log_file_path} results"
            grep -iP "(Accuracy|F1) on (Train|Test) Set" ${log_file_path}
        done
    done
done
