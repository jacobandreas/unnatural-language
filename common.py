from models.sim import SimModel
from models.seq2seq import Seq2SeqModel
import util

from absl import flags
import json
import numpy as np
from pytorch_transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
import torch
import torch.nn.functional as F

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "location of overnight data")
flags.DEFINE_string("dataset", None, "dataset to use")
flags.DEFINE_string("bert_version", "bert-base-uncased", "version of BERT pretrained weights to use")
flags.DEFINE_string("device", "cuda:0", "torch device")
flags.DEFINE_enum("model", "sim", ["sim", "seq2seq"], "model to train")

flags.DEFINE_boolean("lex_features", True, "use lexical features")
flags.DEFINE_boolean("bert_features", True, "use bert features")

flags.DEFINE_integer("max_examples", None, "maximum number of examples to read")
flags.DEFINE_float("train_frac", 1, "fraction of examples to train on")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("train_iters", 1000, "number of training iterations")

flags.DEFINE_string("write_vocab", "vocab.json", "")
flags.DEFINE_string("write_utt_reps", "utt_reps.npy", "")
flags.DEFINE_string("write_utts", "utts.json", "")
flags.DEFINE_string("write_lfs", "lfs.json", "")
flags.DEFINE_string("write_model", "model.p", "")

def _device():
    return torch.device(FLAGS.device)

def _representer(vocab):
    tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_version)
    representer = BertModel.from_pretrained(FLAGS.bert_version, output_hidden_states=True).to(_device())

    def represent(utt):
        out = []

        if FLAGS.bert_features:
            utt_enc = torch.tensor([tokenizer.encode(utt)]).to(_device())
            with torch.no_grad():
                _, _, hiddens = representer(utt_enc)
                word_rep = hiddens[0].mean(dim=1)
                seq_rep = hiddens[-1].mean(dim=1)
            out.append(F.normalize(word_rep, dim=1))
            out.append(F.normalize(seq_rep, dim=1))

        if FLAGS.lex_features:
            utt_lex = np.zeros((1, len(vocab)), dtype=np.float32)
            for word in util.word_tokenize(utt):
                if word in vocab:
                    utt_lex[0, vocab[word]] = 1
            out.append(F.normalize(torch.tensor(utt_lex).to(_device()), dim=1))

        if len(out) == 1:
            return out[0]
        else:
            return torch.cat(out, dim=1)

    return represent

def _embedder(vocab):
    tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_version)
    representer = BertModel.from_pretrained(FLAGS.bert_version, output_hidden_states=True).to(_device())

    def embed(utt):
        assert(FLAGS.bert_features)
        utt_enc = torch.tensor([tokenizer.encode(utt)]).to(_device())
        with torch.no_grad():
            _, _, hiddens = representer(utt_enc)
            return torch.cat([hiddens[0], hiddens[-1]], dim=2)

    return embed

def _model():
    with open(FLAGS.write_vocab) as f:
        vocab = json.load(f)
    with open(FLAGS.write_utts) as f:
        utts = json.load(f)
    with open(FLAGS.write_lfs) as f:
        lfs = json.load(f)
    utt_reps = torch.tensor(np.load(FLAGS.write_utt_reps)).to(_device())
    representer = _representer(vocab)
    embedder = _embedder(vocab)
    test_emb = embedder("test")

    if FLAGS.model == "sim":
        model = SimModel(utt_reps, utts, lfs, representer, _device())
    elif FLAGS.model == "seq2seq":
        model = Seq2SeqModel(test_emb.shape[2], vocab, utts, lfs, embedder, _device())
    else:
        assert False
    return model
