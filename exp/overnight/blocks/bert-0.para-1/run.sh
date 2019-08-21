#!/bin/sh

. ./env.sh

SEMPRE_DIR="$HOME/code/sempre"
BASE_DIR=$(dirname $(readlink -e $0))

python3 ../../../../preproc.py \
    --data_dir=$DATA_DIR \
    --dataset=$DATASET \
    --lex_features=$LEX_FEATURES \
    --bert_features=$BERT_FEATURES \
    &> preproc.log

python3 ../../../../train.py \
    --model=$MODEL \
    --data_dir=$DATA_DIR \
    --dataset=$DATASET \
    --lex_features=$LEX_FEATURES \
    --bert_features=$BERT_FEATURES \
    --train_on_paraphrase=$PARAPHRASE \
    --max_examples=2000 \
    --train_frac=0.8 \
    &> train.log

cd $SEMPRE_DIR
./run @mode=overnight \
    @domain=$DATASET \
    -Builder.parser WrappedParser \
    -WrappedParser.parser $BASE_DIR/parse.sh \
    -trainFrac 0 \
    -devFrac 0.2 \
    -maxExamples train:2000 test:500 \
    &> $BASE_DIR/predict.log
cd $BASE_DIR

#python3 ../../../../predict.py \
#     --model=$MODEL \
#     --data_dir=$DATA_DIR \
#     --dataset=$DATASET \
#     --lex_features=$LEX_FEATURES \
#     --bert_features=$BERT_FEATURES \
#     --train_on_paraphrase=$PARAPHRASE \
#    &> predict.log

