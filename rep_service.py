#!/usr/bin/env python3

import common
from common import _representer

from absl import app, flags
import numpy as np
import os
import sexpdata
import sys

FLAGS = flags.FLAGS

def main(argv):
    rep = _representer(vocab=dict())
    for line in sys.stdin:
        utterance = line.strip()
        pred_rep = rep(utterance)[0, :].detach().cpu().numpy().tolist()
        pred_rep = ["%.4f" % v for v in pred_rep]
        print(" ".join(pred_rep))
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    app.run(main)
