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

    train_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.train.examples".format(FLAGS.dataset))
    with open(train_file) as f:
        train_str = f.read()
        train_data = sexpdata.loads("({})".format(train_str))
    for datum in train_data:
        real = datum[1][1]
        fake = datum[2][1]
        fake_utts.append(fake)
        real_utts.append(real)

    #train_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.groups".format(FLAGS.dataset))
    #last_fake = None
    #with open(train_file) as f:
    #    for line in f:
    #        if line.startswith("original"):
    #            last_fake = line.split("-")[1].strip()
    #            print("fake", last_fake)
    #            continue
    #        line = line.split(",")[0]
    #        real = line.split("-")[1].strip()
    #        print("real", line)
    #        if real in test_real:
    #            print("test")
    #            continue
    #        fake_utts.append(last_fake)
    #        real_utts.append(real)

    model.train(real_utts, fake_utts)
    model.save(FLAGS.write_model)

if __name__ == "__main__":
    app.run(main)
