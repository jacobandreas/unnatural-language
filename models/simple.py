from util import lf_detokenize

from absl import flags
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

FLAGS = flags.FLAGS

def dist_l2(query, target):
    return torch.norm(query - target, dim=1)

def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2

class Model(object):
    pass

class Scorer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, size))
        self.query_score = nn.Linear(size, 1)
        self.pred_score = nn.Bilinear(size, size, 1)
        #self.query_proj = nn.Linear(size, size, bias=False)
        #self.target_proj = nn.Linear(size, size, bias=False)

    def forward(self, query, target):
        return (self.query_score(query) + self.pred_score(query, target)).squeeze(1)
        #return (self.query_score(query)).squeeze(1) + (self.query_proj(query) * self.target_proj(target)).sum(dim=1)
        #return (query * target * self.weight.expand_as(query)).sum(dim=1)

class DistScorer(nn.Module):
    def __init__(self, size):
        super().__init__()

    def forward(self, query, target):
        return dist_cos(query, target)

class SimpleModel(Model):
    def __init__(self, learned_scorer, utt_reps, utts, lfs, representer, device):
        self.utt_reps = utt_reps
        self.utts = utts
        self.lfs = lfs
        self.representer = representer
        self.device = device

        n_features = self.utt_reps.shape[1]
        if learned_scorer:
            print("built scorer", learned_scorer)
            self.scorer = Scorer(n_features)
        else:
            print("built dist", learned_scorer)
            self.scorer = DistScorer(n_features)
        self.scorer.to(device)

    def train(self, real_utts, fake_utts):
        if isinstance(self.scorer, DistScorer):
            return

        real_reps = [self.representer(utt).squeeze(0) for utt in tqdm(real_utts)]
        fake_reps = [self.representer(utt).squeeze(0) for utt in tqdm(fake_utts)]

        opt = optim.Adam(self.scorer.parameters(), lr=0.001)
        total_loss = 0

        for i in range(5000):
            if (i+1) % 100 == 0:
                print("{:.3f}".format(total_loss / 100))
                total_loss = 0

            true_indices = np.random.randint(len(real_reps), size=100)
            false_indices = np.random.randint(len(real_reps), size=100)
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

        #print(self.utts[best])

        nbest = scores.argsort().cpu().numpy()
        for n, i in enumerate(nbest):
            gold = "*" if self.lfs[i] == gold_lf else " "
            if n < 10 or gold == "*":
                print("{:4d} {:4d} {} {:0.3f}".format(n, i, gold, scores[i]), self.utts[i])
        print(best)

        return self.lfs[best]
