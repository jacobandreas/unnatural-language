#!/bin/sh

BASEDIR=$(dirname $(readlink -e $0))
cd $BASEDIR
. ./env.sh
python3 ../../../../predict_interactive.py \
     --model=$MODEL \
     --data_dir=$DATA_DIR \
     --dataset=$DATASET \
     --lex_features=$LEX_FEATURES \
     --bert_features=$BERT_FEATURES \
     --train_on_paraphrase=$PARAPHRASE
