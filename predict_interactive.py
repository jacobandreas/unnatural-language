#!/usr/bin/env python3

import common
from common import _model

from absl import app, flags
import numpy as np
import os
import sexpdata
import sys

FLAGS = flags.FLAGS

def main(argv):
    model = _model()
    model.load(FLAGS.write_model)
    for line in sys.stdin:
        utterance = line.strip()
        pred_lf = model.predict(utterance, None)
        print(pred_lf, file=sys.stderr)
        print(pred_lf)
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    app.run(main)
