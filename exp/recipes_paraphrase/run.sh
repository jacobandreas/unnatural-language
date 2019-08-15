#!/bin/sh

MODEL="simple"
DATA_DIR="$HOME/data/overnight"
DATASET="recipes"

LEX_FEATURES=1
BERT_FEATURES=0
PARAPHRASE=1

python3 ../../preproc.py \
    --data_dir=$DATA_DIR \
    --dataset=$DATASET \
    --lex_features=$LEX_FEATURES \
    --bert_features=$BERT_FEATURES \
    &> preproc.log

python3 ../../train.py \
    --model=$MODEL \
    --data_dir=$DATA_DIR \
    --dataset=$DATASET \
    --lex_features=$LEX_FEATURES \
    --bert_features=$BERT_FEATURES \
    --train_on_paraphrase=$PARAPHRASE \
    &> train.log

python3 ../../predict.py \
     --model=$MODEL \
     --data_dir=$DATA_DIR \
     --dataset=$DATASET \
     --lex_features=$LEX_FEATURES \
     --bert_features=$BERT_FEATURES \
     --train_on_paraphrase=$PARAPHRASE \
    &> predict.log

