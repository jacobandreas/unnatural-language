#!/usr/bin/env python3

import common
from common import _model

from absl import app, flags
import numpy as np
import os
import sexpdata

FLAGS = flags.FLAGS

def main(argv):
    model = _model()
    model.load(FLAGS.write_model)

    test_file = os.path.join(FLAGS.data_dir, "data", "{}.paraphrases.test.examples".format(FLAGS.dataset))
    with open(test_file) as f:
        data_str = f.read()
        data = sexpdata.loads("({})".format(data_str))

    predictions = []
    scores = []
    for datum in data:
        utterance = datum[1][1]
        lf = sexpdata.dumps(datum[3][1]).replace("\\.", ".")

        print()
        print(utterance)
        pred_lf = model.predict(utterance, lf)
        predictions.append(pred_lf)
        scores.append(pred_lf == lf)
        print(pred_lf == lf)

    print(np.mean(scores))

if __name__ == "__main__":
    app.run(main)
