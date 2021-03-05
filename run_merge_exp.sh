#!/bin/bash

export logdir_root=merged_run
export TASK=MERGED
export time=`date +"%Y-%m-%d-%T"`

# bert-base-chinese
python run_squad.py   \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --train_file data/DRCD_CMRC2018/train.json \
  --predict_file data/DRCD/DRCD_dev.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_bert-base-chinese-${time} \
  --log_dir ${logdir_root}/merge_bert-base-chinese-${time} \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --logging_steps 50 \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_bert-base-chinese-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/DRCD/DRCD_test.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_bert-base-chinese-${time} \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_bert-base-chinese-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/CMRC2018/cmrc2018_dev.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_bert-base-chinese-${time} \
  --threads 4 \
  --overwrite_cache
  
bert-wwm-ext
python run_squad.py   \
  --model_type bert \
  --model_name_or_path hfl/chinese-bert-wwm-ext \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --train_file data/DRCD_CMRC2018/train.json \
  --predict_file data/DRCD/DRCD_dev.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_chinese-bert-wwm-ext-${time} \
  --log_dir ${logdir_root}/merge_chinese-bert-wwm-ext-${time} \
  --gradient_accumulation_steps 4 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_chinese-bert-wwm-ext-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/DRCD/DRCD_test.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_chinese-bert-wwm-ext-${time} \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_chinese-bert-wwm-ext-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/CMRC2018/cmrc2018_dev.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_chinese-bert-wwm-ext-${time} \
  --threads 4 \
  --overwrite_cache
  

# roberta_chinese_clue_large
python run_squad.py   \
  --model_type bert \
  --model_name_or_path  clue/roberta_chinese_clue_large \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --train_file data/DRCD_CMRC2018/train.json \
  --predict_file data/DRCD/DRCD_dev.json \
  --per_gpu_train_batch_size 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_roberta_chinese_clue_large-${time} \
  --log_dir ${logdir_root}/merge_roberta_chinese_clue_large-${time} \
  --gradient_accumulation_steps 16 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_roberta_chinese_clue_large-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/DRCD/DRCD_test.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_roberta_chinese_clue_large-${time} \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_roberta_chinese_clue_large-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/CMRC2018/cmrc2018_dev.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_roberta_chinese_clue_large-${time} \
  --threads 4 \
  --overwrite_cache
  
  
# chinese-roberta-wwm-ext-large
python run_squad.py   \
  --model_type bert \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --task ${TASK} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --train_file data/DRCD_CMRC2018/train.json \
  --predict_file data/DRCD/DRCD_dev.json \
  --per_gpu_train_batch_size 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_chinese-roberta-wwm-ext-large-${time} \
  --log_dir ${logdir_root}/merge_chinese-roberta-wwm-ext-large-${time} \
  --gradient_accumulation_steps 16 \
  --logging_steps 50 \
  --overwrite_output_dir \
  --threads 4 \
  --overwrite_cache
  
  python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_chinese-roberta-wwm-ext-large-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file DRCD/DRCD_test.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_chinese-roberta-wwm-ext-large-${time} \
  --threads 4 \
  --overwrite_cache
  
python run_squad.py   \
  --model_type bert \
  --model_name_or_path out/merge_chinese-roberta-wwm-ext-large-${time} \
  --task ${TASK} \
  --do_eval \
  --do_lower_case \
  --do_convert_to_simpified_chinese \
  --predict_file data/CMRC2018/cmrc2018_dev.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir out/merge_chinese-roberta-wwm-ext-large-${time} \
  --threads 4 \
  --overwrite_cache
  