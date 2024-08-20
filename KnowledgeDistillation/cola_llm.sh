#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"


export PYTHONPATH=$(pwd)
# Fine-tune teacher model.
# Hyperparameter can refer to Deberta-v3's settings. They are stored at ./experiments/glue

echo "-------------------"
echo "TEACHER CoLA LARGE"
echo "-------------------"

python a0a_run_teachermodel.py \
  --model_config ./experiments/glue/config.json  \
    --tag deberta-v3-large \
    --do_train \
    --max_seq_len 64 \
    --task_name CoLA \
    --data_dir ./tmp/DeBERTa/glue/CoLA \
    --init_model deberta-v3-large \
    --output_dir ./tmp/results/teacher-v3-large/CoLA   \
    --num_train_epochs 10 \
    --fp16 True \
    --warmup 50 \
    --learning_rate 5.5e-6 \
    --train_batch_size 32 \
    --cls_drop_out 0.1


# Gether Datasets for training
python ./gather_json/cola_tsv_reader.py 
# Build Sense Dictionary
python a1_getsingle.py --json_file ./gather_json/cola_all.json --k 5 --output_keyword ./cola --teacher_ckpt_path ./tmp/results/teacher-v3-large/CoLA/pytorch.model-001072.bin
python a2_clean_cluster.py --output_keyword ./cola --k 5 



echo "-------------------"
echo "REPLACEMENT"
echo "-------------------"

python a0a_run_teachermodel.py \
  --model_config ./experiments/glue/config.json  \
    --tag deberta-v3-large \
    --do_eval \
    --max_seq_len 64 \
    --task_name CoLA \
    --data_dir ./tmp/DeBERTa/glue/CoLA \
    --init_model deberta-v3-large \
    --output_dir ./tmp/results/llm_replace/CoLA   \
    --fp16 True \
    --cluster_path ./cola_5.pkl \
    ---teacher_ckpt_path ./tmp/results/teacher-v3-large/CoLA/pytorch.model-001072.bin
