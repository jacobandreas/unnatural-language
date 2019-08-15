#!/usr/bin/env python3

import common
from common import _model

from absl import app, flags
import os
import sexpdata

FLAGS = flags.FLAGS

def main(argv):
    model = _model()
    
    test_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.test.examples".format(FLAGS.dataset))
    with open(test_file) as f:
        test_str = f.read()
        test_data = sexpdata.loads("({})".format(test_str))
    test_fake = set()
    for datum in test_data:
        real = datum[1][1]
        fake = datum[2][1]
        test_fake.add(fake)

    train_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.groups".format(FLAGS.dataset))
    fake_utts = []
    real_utts = []
    last_fake = None
    with open(train_file) as f:
        for line in f:
            if line.startswith("original"):
                last_fake = line.split("-")[1].strip()
                print("fake", last_fake)
                continue
            if last_fake in fake:
                print("test")
                continue
            line = line.split(",")[0]
            line = line.split("-")[1].strip()
            fake_utts.append(last_fake)
            real_utts.append(line)
            print("real", line)

    model.train(real_utts, fake_utts)
    model.save(FLAGS.write_model)

if __name__ == "__main__":
    app.run(main)
