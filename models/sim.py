from util import lf_detokenize

from absl import flags
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
import sys

FLAGS = flags.FLAGS
flags.DEFINE_enum("sim__scorer", "bilinear", ["", "linear", "bilinear", "rnn"], "scoring function to use")
flags.DEFINE_integer("sim__hidden_size", 1024, "size of hidden state for similarity model")
flags.DEFINE_boolean("sim__supervised", False, "train on examples")

def dist_l2(query, target):
    return torch.norm(query - target, dim=1)

def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2

class Scorer(nn.Module):
    def __init__(self, size, emb_size):
        super().__init__()
        self.query_score = nn.Linear(size, 1)
        if FLAGS.sim__scorer == "linear":
            self.pred_score = nn.Linear(size, 1)
        elif FLAGS.sim__scorer == "bilinear":
            self.pred_score = nn.Bilinear(size, size, 1)
        elif FLAGS.sim__scorer == "rnn":
            assert(FLAGS.bert_features)
            self.encoder_query = nn.LSTM(emb_size, FLAGS.sim__hidden_size, 1, bidirectional=True)
            self.encoder_target = nn.LSTM(emb_size, FLAGS.sim__hidden_size, 1, bidirectional=True)
        else:
            assert False

    def forward(self, query_rep, target_rep):
        if FLAGS.sim__scorer == "linear":
            return (self.query_score(query) + self.pred_score(query * target * np.sqrt(query.shape[1]))).squeeze(1)
        elif FLAGS.sim__scorer == "bilinear":
            return (self.query_score(query) + self.pred_score(query, target)).squeeze(1)
        elif FLAGS.sim__scorer == "rnn":
            _, (query_enc, _) = self.encoder_query(query_rep)
            _, (target_enc, _) = self.encoder_target(target_rep)
            return (query_enc * target_enc).sum(dim=2).sum(dim=0)
        else:
            assert False

class DistScorer(nn.Module):
    def __init__(self, size):
        super().__init__()

    def forward(self, query, target):
        return dist_cos(query, target)

class SimModel(object):
    def __init__(self, utt_reps, utt_embs, utts, lfs, representer, embedder, device):
        self.utt_reps = utt_reps
        self.utt_embs = utt_embs
        self.utts = utts
        self.lfs = lfs
        self.representer = representer
        self.embedder = embedder
        self.device = device

        n_features = utt_reps.shape[-1]
        n_emb_features = utt_embs.shape[-1]
        if FLAGS.sim__supervised:
            self.scorer = Scorer(n_features, n_emb_features)
        else:
            self.scorer = DistScorer(n_features)
        self.scorer.to(device)

    def _pad_cat(self, reps):
        max_len = max(rep.shape[1] for rep in reps)
        data = torch.zeros((max_len, len(reps), reps[0].shape[2])).to(self.device)
        for i, rep in enumerate(reps):
            data[:rep.shape[1], i, :] = rep[0, ...]
        return data

    def train(self, real_utts, fake_utts, lfs):
        self.train_cls(real_utts, fake_utts, lfs)

    #def train_cls(self, real_utts, fake_utts, lfs):
    #    labels = [self.lfs.index(l) for l in lfs]
    #    opt = optim.Adam(self.scorer.parameters(), lr=0.0003)
    #    real_reps = [self.representer(utt) for utt in tqdm(real_utts)]
    #    real_embs = [self.embedder(utt) for utt in tqdm(real_utts)]
    #    total_loss = 0
    #    for i in range(FLAGS.train_iters):
    #        if (i+1) % 100 == 0:
    #            print("{:.3f}".format(total_loss / 100), file=sys.stderr)
    #        batch_indices = np.random.randint(len(real_reps), size=FLAGS.batch_size)
    #        if FLAGS.sim__scorer = "rnn":
    #            query = self

    def train_metric(self, real_utts, fake_utts, lfs):
        if isinstance(self.scorer, DistScorer):
            return

        real_reps = [self.representer(utt) for utt in tqdm(real_utts)]
        fake_reps = [self.representer(utt) for utt in tqdm(fake_utts)]
        real_embs = [self.embedder(utt) for utt in tqdm(real_utts)]
        fake_embs = [self.embedder(utt) for utt in tqdm(fake_utts)]

        opt = optim.Adam(self.scorer.parameters(), lr=0.001)
        total_loss = 0

        for i in range(FLAGS.train_iters):
            if (i+1) % 100 == 0:
                print("{:.3f}".format(total_loss / 100), file=sys.stderr)
                total_loss = 0

            true_indices = np.random.randint(len(real_reps), size=FLAGS.batch_size)
            false_indices = np.random.randint(len(real_reps), size=FLAGS.batch_size)

            if FLAGS.sim__scorer == "rnn":
                pred_reps = self._pad_cat([real_embs[i] for i in true_indices])
                true_reps = self._pad_cat([fake_embs[i] for i in true_indices])
                false_reps = self._pad_cat([fake_embs[i] for i in false_indices])
            else:
                pred_reps = torch.cat([real_reps[i] for i in true_indices], dim=0)
                true_reps = torch.cat([fake_reps[i] for i in true_indices], dim=0)
                false_reps = torch.cat([fake_reps[i] for i in false_indices], dim=0)

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
        if FLAGS.sim__scorer == "rnn":
            rep = self.embedder(utt).squeeze(0).unsqueeze(1).expand(-1, self.utt_embs.shape[1], -1)
            scores = self.scorer(self.utt_embs, rep)
        else:
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

