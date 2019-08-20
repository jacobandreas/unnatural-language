#!/bin/sh

MODEL="simple"
DATA_DIR="$HOME/data/overnight"

DATASETS="basketball blocks calendar housing publications recipes restaurants socialnetwork"
#DATASETS="basketball"

basedir=$(dirname "$0")
cd $basedir

for dataset in $DATASETS
do
  for bert in 0 1
  do
    for para in 0 1
    do
      experiment_dir="$dataset/bert-$bert.para-$para"
      mkdir -p $experiment_dir
      cd $experiment_dir
      cat << EOF > env.sh
MODEL="$MODEL"
DATA_DIR="$HOME/data/overnight"
DATASET="$dataset"
LEX_FEATURES=1
BERT_FEATURES="$bert"
PARAPHRASE="$para"
EOF
      cp ../../run.template.sh run.sh
      cp ../../parse.template.sh parse.sh
      chmod +x run.sh parse.sh
      ./run.sh
      cd ../..
    done
  done
done
