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
flags.DEFINE_string("write_word_reps", "word_reps.npy", "")
flags.DEFINE_string("write_utts", "utts.json", "")
flags.DEFINE_string("write_lfs", "lfs.json", "")
flags.DEFINE_string("write_model", "model.p", "")

BERT_SINGLETONS = {}

def _ensure_bert():
    if "tokenizer" not in BERT_SINGLETONS:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_version)
        BERT_SINGLETONS["tokenizer"] = tokenizer

    if "representer" not in BERT_SINGLETONS:
        representer = BertModel.from_pretrained(FLAGS.bert_version, output_hidden_states=True).to(_device())
        BERT_SINGLETONS["representer"] = representer

    return BERT_SINGLETONS["tokenizer"], BERT_SINGLETONS["representer"]

def _device():
    return torch.device(FLAGS.device)

def _sent_representer(vocab):
    if FLAGS.bert_features:
        tokenizer, representer = _ensure_bert()
    else:
        tokenizer = representer = None

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
            return out[0].detach()
        else:
            return torch.cat(out, dim=1).detach()

    return represent

def _word_representer(vocab):
    tokenizer, representer = _ensure_bert()

    def represent(utt):
        out = []
        utt_words = util.word_tokenize(utt)
        utt_enc = torch.tensor([tokenizer.encode(utt)]).to(_device())
        if FLAGS.bert_features:
            with torch.no_grad():
                _, _, hiddens = representer(utt_enc)
            out.append(hiddens[0])
            out.append(hiddens[-1])
        
        if FLAGS.lex_features:
            one_hot = torch.zeros(1, utt_enc.shape[1], len(vocab))
            j = 0
            for i in range(len(utt_enc)):
                dec = tokenizer.decode(utt_enc[i])
                if not dec.startswith("##"):
                    word = utt_words[j]
                    if word in vocab:
                        one_hot[0, i, vocab[word]] = 1
                    j += 1
            one_hot = one_hot.to(_device())
            out.append(one_hot)

        if len(out) == 1:
            return out[0].detach()
        else:
            return torch.cat(out, dim=2).detach()

    return represent

def _model():
    with open(FLAGS.write_vocab) as f:
        vocab = json.load(f)
    with open(FLAGS.write_utts) as f:
        utts = json.load(f)
    with open(FLAGS.write_lfs) as f:
        lfs = json.load(f)
    utt_reps = torch.tensor(np.load(FLAGS.write_utt_reps)).to(_device())
    word_reps = torch.tensor(np.load(FLAGS.write_word_reps).astype(np.float32)).to(_device())
    representer = _sent_representer(vocab)
    embedder = _word_representer(vocab)

    if FLAGS.model == "sim":
        model = SimModel(utt_reps, word_reps, utts, lfs, representer, embedder, _device())
    elif FLAGS.model == "seq2seq":
        model = Seq2SeqModel(word_reps.shape[2], vocab, utt_reps, utts, lfs, embedder, _device())
    else:
        assert False
    return model
