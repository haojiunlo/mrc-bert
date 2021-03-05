# Data:
    - https://github.com/DRCKnowledgeTeam/DRCD
    - https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip
    
# Usage
* example

```bash
export data_dir=data/DRCD
export logdir_root=runs
export TASK=DRCD
export time=`date +"%Y-%m-%d-%T"`

python run_squad.py \
        --model_type bert \
        --model_name_or_path bert-base-chinese \
        --task ${TASK} \
        --do_train \
        --do_eval \
        --do_lower_case \
        --do_convert_to_simpified_chinese \
        --data_dir ${data_dir} \
        --train_file DRCD_training.json \
        --predict_file DRCD_dev.json \
        --per_gpu_train_batch_size 8 \
        --learning_rate 3e-5 \
        --num_train_epochs 2.0 \
        --max_seq_length 512 \
        --doc_stride 128 \
        --output_dir out/DRCD_bert-base-chinese-${time} \
        --log_dir ${logdir_root}/DRCD_bert-base-chinese-${time} \
        --overwrite_output_dir \
        --gradient_accumulation_steps 4 \
        --logging_steps 50 \
        --threads 4 \
        --overwrite_cache
```

# Result
|MODEL|DEV_F1|DEV_EM|TEST_F1|TEST_EM|params|
|---|---|---|---|---|---|
|BERT-wwm-ext|93.51|88.37|92.28|86.55|gradient_accumulation_steps=4, learning_rate=3e-05, max_seq_length=512, num_train_epochs=2.0, per_gpu_train_batch_size=8, warmup_proportion=0.1|
|BERT-base-chinese|92.21|86.46|91.52|85.97|gradient_accumulation_steps=4, learning_rate=3e-05, max_seq_length=512, num_train_epochs=2.0, per_gpu_train_batch_size=8, warmup_proportion=0.1|
|RoBerta-wwm-ext-large|95.28|90.47|94.93|90.24|gradient_accumulation_steps: 16, learning_rate: 3e-05, max_seq_length: 512, num_train_epochs: 2.0, per_gpu_train_batch_size: 2, warmup_proportion: 0.1|
|RoBerta-large-clue t2s|94.81|90.38|94.56|89.87|gradient_accumulation_steps: 16, learning_rate: 3e-05, max_seq_length: 512, num_train_epochs: 2.0, per_gpu_train_batch_size: 2, warmup_proportion: 0.1|


# Reference
https://github.com/CLUEbenchmark/CLUE  
https://github.com/huggingface/transformers
