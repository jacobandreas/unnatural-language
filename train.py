#!/usr/bin/env python3

import common
from common import _model

from absl import app, flags
import os
import sexpdata

FLAGS = flags.FLAGS

def main(argv):
    model = _model()
    
    fake_utts = []
    real_utts = []
    lfs = []

    train_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.train.examples".format(FLAGS.dataset))
    with open(train_file) as f:
        train_str = f.read()
        train_data = sexpdata.loads("({})".format(train_str))

    num_train = len(train_data)
    if FLAGS.max_examples is not None:
        num_train = min(num_train, FLAGS.max_examples)
    num_train = int(num_train * FLAGS.train_frac)
    train_data = train_data[:num_train]

    for datum in train_data:
        real = datum[1][1]
        fake = datum[2][1]
        lf = sexpdata.dumps(datum[3][1]).replace("\\.", ".")
        fake_utts.append(fake)
        real_utts.append(real)
        lfs.append(lf)

    model.train(real_utts, fake_utts, lfs)
    model.save(FLAGS.write_model)

if __name__ == "__main__":
    app.run(main)
