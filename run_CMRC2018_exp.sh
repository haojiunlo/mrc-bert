#!/bin/bash

export data_dir=data/CMRC2018/
export logdir_root=cmrc2018_runs
export TASK=CMRC2018
export time=`date +"%Y-%m-%d-%T"`

# bert-base-chinese
python run_squad.py   \
  --model_type bert \
  --model_name_or_path  bert-base-chinese \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ${data_dir} \
  --train_file cmrc2018_train.json \
  --predict_file cmrc2018_dev.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/CMRC2018_bert-base-chinese-${time} \
  --log_dir ${logdir_root}/CMRC2018_bert-base-chinese-${time} \
  --gradient_accumulation_steps 4 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --threads 4

# chinese-bert-wwm-ext
python run_squad.py   \
  --model_type bert \
  --model_name_or_path hfl/chinese-bert-wwm-ext \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ${data_dir} \
  --train_file cmrc2018_train.json \
  --predict_file cmrc2018_dev.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/CMRC2018_bert-wwm-ext-${time} \
  --log_dir ${logdir_root}/CMRC2018_bert-wwm-ext-${time} \
  --gradient_accumulation_steps 4 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --threads 4
  
# chinese-roberta-wwm-ext-large
python run_squad.py   \
  --model_type bert \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ${data_dir} \
  --train_file cmrc2018_train.json \
  --predict_file cmrc2018_dev.json \
  --per_gpu_train_batch_size 2 \
  --learning_rate 2.5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/CMRC2018_chinese-roberta-wwm-ext-large-${time} \
  --log_dir ${logdir_root}/CMRC2018_chinese-roberta-wwm-ext-large-${time} \
  --gradient_accumulation_steps 16 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --threads 4

# clue/roberta_chinese_clue_large
python run_squad.py   \
  --model_type bert \
  --model_name_or_path clue/roberta_chinese_clue_large \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ${data_dir} \
  --train_file cmrc2018_train.json \
  --predict_file cmrc2018_dev.json \
  --per_gpu_train_batch_size 2 \
  --learning_rate 2.5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/CMRC2018_roberta_chinese_clue_large-${time} \
  --log_dir ${logdir_root}/CMRC2018_roberta_chinese_clue_large-${time} \
  --overwrite_output_dir \
  --gradient_accumulation_steps 16 \
  --logging_steps 50 \
  --threads 4
  