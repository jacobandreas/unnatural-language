#!/usr/bin/env python3

import common
from common import _representer, _device
from util import lf_tokenize

from absl import app, flags
import json
import numpy as np
import os
import torch
from tqdm import tqdm

FLAGS = flags.FLAGS

def main(argv):
    canonical_utt_file = os.path.join(FLAGS.data_dir, "genovernight.out", FLAGS.dataset, "utterances_formula.tsv")

    vocab = {}
    with open(canonical_utt_file) as f:
        for line in tqdm(f):
            utt, lf = line.strip().split("\t")
            words = utt.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)

    representer = _representer(vocab)

    utt_reps = []
    utts = []
    lfs = []
    with open(canonical_utt_file) as f:
        for line in tqdm(f):
            utt, lf = line.strip().split("\t")
            utt_reps.append(representer(utt).squeeze(0).detach().cpu().numpy())
            utts.append(utt)
            lfs.append(lf)

    with open(FLAGS.write_vocab, "w") as f:
        json.dump(vocab, f)
    np.save(FLAGS.write_utt_reps, utt_reps)
    with open(FLAGS.write_utts, "w") as f:
        json.dump(utts, f)
    with open(FLAGS.write_lfs, "w") as f:
        json.dump(lfs, f)

if __name__ == "__main__":
    app.run(main)
