#!/usr/bin/env bash

source ./source.conf

function create_data()
{
export BERT_BASE_DIR=models_pretrain/albert_tiny_250k
$python create_pretraining_data.py \
    --do_whole_word_mask=True \
    --input_file=data/news_zh_1.txt \
    --output_file=data/tf_news_2016_zh_raw_news2016zh_1.tfrecord \
    --vocab_file=$BERT_BASE_DIR/vocab.txt 
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=51 \
    --masked_lm_prob=0.10
}


function pretrain()
{
$python run_pretraining.py \
        --input_file=./data/tf*.tfrecord  \
        --output_dir=my_new_model_path \
        --do_train=True \
        --do_eval=True \
        --bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json \
        --train_batch_size=32 \
        --max_seq_length=128 \
        --max_predictions_per_seq=51 \
        --num_train_steps=10000 \
        --num_warmup_steps=1000 \
        --learning_rate=0.00176 \
        --save_checkpoints_steps=2000   \
        --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt 
}

function board()
{
    ps aux| grep tensorboard | grep my_new_model_path| awk '{print $2}'| xargs kill -9 
    nohup ~/try/python-2.7.8-tf1.4/bin/python ~/try/python-2.7.8-tf1.4/bin/tensorboard --logdir=./my_new_model_path/ --port=8003 &
}

function main()
{
    board
    create_data
    pretrain
}

main >log/run.log 2>&1
