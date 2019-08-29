from models.torchdec import Vocab, Encoder, Decoder, SimpleAttention, batch_seqs
import util
import sexpdata

from absl import flags
import numpy as np
import torch
from torch import nn, optim
import sys

FLAGS = flags.FLAGS
flags.DEFINE_boolean("seq2seq__pretrained_enc", False, "use pretrained representations for encoder")
flags.DEFINE_boolean("seq2seq__fixed_enc", False, "attend directly to pretrained representation")
flags.DEFINE_integer("seq2seq__hidden_size", 1024, "size of hidden state for seq2seq model")
flags.DEFINE_integer("seq2seq__embed_size", 256, "size of embedding for seq2seq model")

class Implementation(nn.Module):
    def __init__(self, rep_size, vocab, embedder, device):
        super().__init__()
        hidden_size = FLAGS.seq2seq__hidden_size
        embed_size = FLAGS.seq2seq__embed_size
        self.vocab = vocab
        self.embedder = embedder
        self.device = device

        self._emb_cache = {}

        if FLAGS.seq2seq__pretrained_enc:
            if FLAGS.seq2seq__fixed_enc:
                self.encoder = nn.Linear(rep_size, hidden_size)
            else:
                self.encoder = nn.LSTM(rep_size, hidden_size, 1, bidirectional=True)
        else:
            self.encoder = Encoder(vocab, embed_size, hidden_size, 1)

        self.proj = nn.Linear(hidden_size * 2, hidden_size)

        self.decoder = Decoder(
            vocab,
            embed_size,
            hidden_size,
            1,
            copy=False,
            attention=[SimpleAttention(hidden_size, hidden_size)],
        )

    def _encode_pretrained(self, utts_raw):
        emb_words = []
        for utt in utts_raw:
            if utt in self._emb_cache:
                emb = self._emb_cache[utt]
            else:
                emb = self.embedder(utt).permute(1, 0, 2)
                self._emb_cache[utt] = emb
            emb_words.append(emb)
        max_len = max(e.shape[0] for e in emb_words)

        att_toks = np.ones((max_len, len(emb_words)), dtype=np.int64) * self.vocab.pad()
        for i in range(len(emb_words)):
            att_toks[:emb_words[i].shape[0]] = self.vocab.unk()
        att_toks = torch.tensor(att_toks).to(self.device)

        emb_words = [
            torch.cat((
                e,
                torch.zeros(max_len - e.shape[0], 1, e.shape[2]).to(self.device)
            ), dim=0)
            for e in emb_words
        ]
        emb_words = torch.cat(emb_words, dim=1)

        if FLAGS.seq2seq__fixed_enc:
            enc_words = self.encoder(emb_words)
            enc_utt = enc_words.mean(dim=0, keepdim=True)
        else:
            enc_words, (enc_utt, _) = self.encoder(emb_words)
            enc_words = self.proj(enc_words)
            enc_utt = self.proj(torch.cat(enc_utt.split(1), dim=2))
        return enc_words, enc_utt, att_toks

    def _encode_basic(self, utt_data):
        enc_words, (enc_utt, _) = self.encoder(utt_data)
        enc_words = self.proj(enc_words)
        enc_utt = self.proj(torch.cat(enc_utt.split(1), dim=2))
        return enc_words, enc_utt, utt_data

    def forward(self, utts_raw, utt_data, lf_data):
        if FLAGS.seq2seq__pretrained_enc:
            enc_words, enc_utt, att_toks = self._encode_pretrained(utts_raw)
        else:
            enc_words, enc_utt, att_toks = self._encode_basic(utt_data)

        dec_state = (enc_utt, torch.zeros_like(enc_utt))
        logits, *_ = self.decoder(
            dec_state,
            lf_data.shape[0],
            ref_tokens=lf_data,
            att_features=(enc_words,),
            att_tokens=(att_toks,),
        )
        return logits

    def predict(self, utt_raw, utt_data):
        if FLAGS.seq2seq__pretrained_enc:
            enc_words, enc_utt, att_toks = self._encode_pretrained(utt_raw)
        else:
            enc_words, enc_utt, att_toks = self._encode_basic(utt_data)
        dec_state = (enc_utt, torch.zeros_like(enc_utt))
        #preds = self.decoder.beam(
        #    dec_state,
        #    10,
        #    50,
        #    att_features=(enc_words,),
        #    att_tokens=(att_toks,),
        #)
        #preds_out = []
        #for pred in preds:
        #    try:
        #        pred_str = util.lf_detokenize(self.vocab.decode(pred))
        #        parsed = sexpdata.loads(pred_str)
        #    except Exception:
        #        parsed = None
        #    if parsed:
        #        preds_out.append(pred)
        #return preds_out

        preds, _ = self.decoder.sample(
            dec_state,
            100,
            att_features=(enc_words,),
            att_tokens=(att_toks,),
            greedy=True
        )
        return preds

class Seq2SeqModel(nn.Module):
    def __init__(self, rep_size, base_vocab, utt_reps, utts, lfs, embedder, device):
        super().__init__()
        self.utts = utts
        self.lfs = lfs
        self.device = device

        vocab = Vocab()
        for word in base_vocab:
            vocab.add(word)
        for lf in lfs:
            for token in util.lf_tokenize(lf):
                vocab.add(token)
        self.vocab = vocab

        self.implementation = Implementation(rep_size, vocab, embedder, device)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

        self.device = device
        self.to(device)

    def train(self, real_utts, fake_utts, lfs):
        lfs_raw = lfs
        utts_raw = []
        utts_indexed = []
        lfs_indexed = []
        for real, fake, lf in zip(real_utts, fake_utts, lfs):
            real_indexed = self.vocab.encode(util.word_tokenize(real))
            fake_indexed = self.vocab.encode(util.word_tokenize(fake))
            lf_indexed = self.vocab.encode(util.lf_tokenize(lf))
            if real not in utts_raw:
                utts_raw.append(real)
                utts_indexed.append(real_indexed)
                lfs_indexed.append(lf_indexed)
            if fake not in utts_raw:
                utts_raw.append(fake)
                utts_indexed.append(fake_indexed)
                lfs_indexed.append(lf_indexed)

        opt = optim.Adam(self.parameters(), lr=0.0003)
        opt_sched = optim.lr_scheduler.StepLR(opt, step_size=FLAGS.train_iters//2, gamma=0.1)
        total_loss = 0
        self.implementation.train()
        for i in range(FLAGS.train_iters):
            if (i+1) % 10 == 0:
                print("{:.3f}".format(total_loss / 10), file=sys.stderr)
                sys.stderr.flush()
                total_loss = 0

            indices = np.random.randint(len(utts_indexed), size=FLAGS.batch_size)
            batch_utts_raw = [utts_raw[i] for i in indices]
            batch_utts_indexed = [utts_indexed[i] for i in indices]
            batch_utt_data = batch_seqs(batch_utts_indexed).to(self.device)

            lfs = [lfs_indexed[i] for i in indices]
            lf_data = batch_seqs(lfs).to(self.device)
            lf_ctx = lf_data[:-1, :]
            lf_tgt = lf_data[1:, :].view(-1)

            logits = self.implementation(batch_utts_raw, batch_utt_data, lf_ctx)
            logits = logits.view(-1, logits.shape[-1])
            loss = self.loss(logits, lf_tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            opt_sched.step()
            total_loss += loss.item()

        #correct = 0
        #total = 0
        #for utt, lf in zip(real_utts, lfs_raw):
        #    pred_lf = self.predict(utt, lf)
        #    print(len(util.lf_tokenize(pred_lf)))
        #    print(utt)
        #    print(lf)
        #    print(pred_lf)
        #    print(pred_lf == utt)
        #    print()
        #    total += 1
        #    correct += int(lf == pred_lf)
        #print(correct / total)

    def save(self, location):
        torch.save(self.implementation.state_dict(), location)

    def load(self, location):
        self.implementation.load_state_dict(torch.load(location))

    def predict(self, utt, gold_lf):
        self.implementation.eval()
        utt_raw = [utt]
        utt_data = batch_seqs([self.vocab.encode(util.word_tokenize(utt), unk=True)]).to(self.device)
        preds = self.implementation.predict(utt_raw, utt_data)
        if len(preds) == 0:
            return None
        lfs = [util.lf_detokenize(self.vocab.decode(pred)) for pred in preds]
        print("best guess", lfs[0], file=sys.stderr)
        lfs = [lf for lf in lfs if lf in self.lfs]
        if len(lfs) > 0:
            return lfs[0]
        return self.lfs[np.random.randint(len(self.lfs))]
