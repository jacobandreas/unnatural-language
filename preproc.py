#!/usr/bin/env python3

import common
from common import _sent_representer, _word_representer, _device
import util

from absl import app, flags
import json
import numpy as np
import os
import sexpdata
import torch
from tqdm import tqdm

FLAGS = flags.FLAGS

def _pad_cat(reps):
    max_len = max(rep.shape[1] for rep in reps)
    data = np.zeros((max_len, len(reps), reps[0].shape[2]), dtype=np.float32)
    for i, rep in enumerate(reps):
        data[:rep.shape[1], i, :] = rep[0, ...]
    return data

def main(argv):
    canonical_utt_file = os.path.join(FLAGS.data_dir, "genovernight.out", FLAGS.dataset, "utterances_formula.tsv")
    train_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.train.examples".format(FLAGS.dataset))

    vocab = {}
    with open(train_file) as f:
        train_str = f.read()
        train_data = sexpdata.loads("({})".format(train_str))
        for datum in train_data:
            real = datum[1][1]
            words = util.word_tokenize(real)
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
    with open(canonical_utt_file) as f:
        for line in f:
            utt, _ = line.strip().split("\t")
            words = util.word_tokenize(utt)
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)

    sent_representer = _sent_representer(vocab)
    word_representer = _word_representer(vocab)

    sent_reps = []
    word_reps = []
    utts = []
    lfs = []
    with open(canonical_utt_file) as f:
        for line in tqdm(f):
            utt, lf = line.strip().split("\t")
            sent_reps.append(sent_representer(utt).squeeze(0).detach().cpu().numpy())
            word_reps.append(word_representer(utt).detach().cpu().numpy())
            utts.append(utt)
            lfs.append(lf)

    with open(FLAGS.write_vocab, "w") as f:
        json.dump(vocab, f)
    np.save(FLAGS.write_utt_reps, sent_reps)
    np.save(FLAGS.write_word_reps, _pad_cat(word_reps))
    with open(FLAGS.write_utts, "w") as f:
        json.dump(utts, f)
    with open(FLAGS.write_lfs, "w") as f:
        json.dump(lfs, f)

if __name__ == "__main__":
    app.run(main)
