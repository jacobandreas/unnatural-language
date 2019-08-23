from util import lf_detokenize

from absl import flags
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
import sys

FLAGS = flags.FLAGS
flags.DEFINE_enum("sim__scorer", "bilinear", ["", "linear", "bilinear"], "scoring function to use")
flags.DEFINE_boolean("sim__supervised", False, "train on examples")

def dist_l2(query, target):
    return torch.norm(query - target, dim=1)

def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2

class Scorer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.query_score = nn.Linear(size, 1)
        if FLAGS.sim__scorer == "linear":
            self.pred_score = nn.Linear(size, 1)
        elif FLAGS.sim__scorer == "bilinear":
            self.pred_score = nn.Bilinear(size, size, 1)
        else:
            assert False

    def forward(self, query, target):
        if FLAGS.sim__scorer == "linear":
            return (self.query_score(query) + self.pred_score(query * target * np.sqrt(query.shape[1]))).squeeze(1)
        elif FLAGS.sim__scorer == "bilinear":
            return (self.query_score(query) + self.pred_score(query, target)).squeeze(1)
        else:
            assert False

class DistScorer(nn.Module):
    def __init__(self, size):
        super().__init__()

    def forward(self, query, target):
        return dist_cos(query, target)

class SimModel(object):
    def __init__(self, utt_reps, utts, lfs, representer, device):
        self.utt_reps = utt_reps
        self.utts = utts
        self.lfs = lfs
        self.representer = representer
        self.device = device

        n_features = self.utt_reps.shape[1]
        if FLAGS.sim__supervised:
            self.scorer = Scorer(n_features)
        else:
            self.scorer = DistScorer(n_features)
        self.scorer.to(device)

    def train(self, real_utts, fake_utts):
        if isinstance(self.scorer, DistScorer):
            return

        real_reps = [self.representer(utt).squeeze(0) for utt in tqdm(real_utts)]
        fake_reps = [self.representer(utt).squeeze(0) for utt in tqdm(fake_utts)]

        opt = optim.Adam(self.scorer.parameters(), lr=0.001)
        total_loss = 0

        for i in range(FLAGS.train_iters):
            if (i+1) % 100 == 0:
                print("{:.3f}".format(total_loss / 100), file=sys.stderr)
                total_loss = 0

            true_indices = np.random.randint(len(real_reps), size=FLAGS.batch_size)
            false_indices = np.random.randint(len(real_reps), size=FLAGS.batch_size)
            pred_reps = torch.stack([real_reps[i] for i in true_indices])
            true_reps = torch.stack([fake_reps[i] for i in true_indices])
            false_reps = torch.stack([fake_reps[i] for i in false_indices])
            true_dist = self.scorer(true_reps, pred_reps)
            false_dist = self.scorer(false_reps, pred_reps)

            diff = true_dist - false_dist + 1
            loss = torch.max(diff, torch.zeros_like(diff)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    def save(self, location):
        torch.save(self.scorer.state_dict(), location)

    def load(self, location):
        self.scorer.load_state_dict(torch.load(location))

    def predict(self, utt, gold_lf):
        rep = self.representer(utt).expand_as(self.utt_reps)
        scores = self.scorer(self.utt_reps, rep)
        best = scores.argmin()

        nbest = scores.argsort().cpu().numpy()
        for n, i in enumerate(nbest):
            gold = "*" if self.lfs[i] == gold_lf else " "
            if n < 10 or gold == "*":
                print("{:4d} {:4d} {} {:0.3f}".format(n, i, gold, scores[i]), self.utts[i], file=sys.stderr)
        print(best, file=sys.stderr)

        return self.lfs[best]

