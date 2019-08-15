from absl import flags
import json
from models.simple import SimpleModel
import numpy as np
from pytorch_transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
import spacy
import torch
import torch.nn.functional as F

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "location of overnight data")
flags.DEFINE_string("dataset", None, "dataset to use")
flags.DEFINE_string("bert_version", "bert-base-uncased", "version of BERT pretrained weights to use")
flags.DEFINE_string("device", "cuda:0", "torch device")
flags.DEFINE_enum("model", "simple", ["simple"], "model to train")

flags.DEFINE_boolean("lex_features", True, "use lexical features")
flags.DEFINE_boolean("bert_features", True, "use bert features")
flags.DEFINE_boolean("train_on_paraphrase", False, "train a scoring function")

flags.DEFINE_string("write_vocab", "vocab.json", "")
flags.DEFINE_string("write_utt_reps", "utt_reps.npy", "")
flags.DEFINE_string("write_utts", "utts.json", "")
flags.DEFINE_string("write_lfs", "lfs.json", "")
flags.DEFINE_string("write_model", "model.p", "")

def _device():
    return torch.device(FLAGS.device)

def _representer(vocab):
    tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_version)
    representer = BertModel.from_pretrained(FLAGS.bert_version, output_hidden_states=False).to(_device())
    nlp = spacy.load("en_core_web_sm")

    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #representer = GPT2Model.from_pretrained("gpt2").to(_device())

    def represent(utt):
        out = []

        if FLAGS.bert_features:
            utt_enc = torch.tensor([tokenizer.encode(utt)]).to(_device())
            with torch.no_grad():
                utt_rep = representer(utt_enc)[0].mean(dim=1)
            out.append(F.normalize(utt_rep, dim=1))

        if FLAGS.lex_features:
            utt_lex = np.zeros((1, len(vocab)), dtype=np.float32)
            analyzed_utt = nlp(utt)
            for token in analyzed_utt:
                word = token.lemma_
                if word in vocab:
                    utt_lex[0, vocab[word]] = 1
            out.append(F.normalize(torch.tensor(utt_lex).to(_device()), dim=1))

        if len(out) == 1:
            return out[0]
        else:
            return torch.cat(out, dim=1)

    return represent


def _model():
    with open(FLAGS.write_vocab) as f:
        vocab = json.load(f)
    with open(FLAGS.write_utts) as f:
        utts = json.load(f)
    with open(FLAGS.write_lfs) as f:
        lfs = json.load(f)
    utt_reps = torch.tensor(np.load(FLAGS.write_utt_reps)).to(_device())

    assert FLAGS.model == "simple"
    learned_scorer = FLAGS.train_on_paraphrase
    return SimpleModel(learned_scorer, utt_reps, utts, lfs, _representer(vocab), _device())
