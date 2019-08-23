from models.torchdec import Vocab, Encoder, Decoder, SimpleAttention, batch_seqs
import util

from absl import flags
import numpy as np
import torch
from torch import nn, optim
import sys

FLAGS = flags.FLAGS
flags.DEFINE_boolean("seq2seq__pretrained_enc", False, "use pretrained representations for encoder")
flags.DEFINE_integer("seq2seq__hidden_size", 1024, "size of hidden state for seq2seq model")
flags.DEFINE_integer("seq2seq__embed_size", 256, "size of embedding for seq2seq model")

class Seq2SeqModel(nn.Module):
    def __init__(self, rep_size, base_vocab, utts, lfs, embedder, device):
        super().__init__()
        self.utts = utts
        self.lfs = lfs
        self.embedder = embedder
        self.device = device

        hidden_size = FLAGS.seq2seq__hidden_size
        embed_size = FLAGS.seq2seq__embed_size

        vocab = Vocab()
        for word in base_vocab:
            vocab.add(word)
        for lf in lfs:
            for token in util.lf_tokenize(lf):
                vocab.add(token)
        self.vocab = vocab

        if FLAGS.seq2seq__pretrained_enc:
            self.proj = nn.Linear(rep_size, hidden_size)
            self.rep_size = rep_size
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

        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

        self.device = device
        self.to(device)

    def _encode_pretrained(self, utts_raw):
        enc_words = [self.embedder(utt).permute(1, 0, 2) for utt in utts_raw]
        max_len = max(e.shape[0] for e in enc_words)

        att_toks = np.ones((max_len, len(enc_words)), dtype=np.int64) * self.vocab.pad()
        for i in range(len(enc_words)):
            att_toks[:len(enc_words)] = self.vocab.unk()
        att_toks = torch.tensor(att_toks).to(self.device)

        enc_words = [
            torch.cat((
                e,
                torch.zeros(max_len - e.shape[0], 1, e.shape[2]).to(self.device)
            ), dim=0)
            for e in enc_words
        ]
        enc_words = torch.cat(enc_words, dim=1)
        enc_words = self.proj(enc_words)
        enc_utt = enc_words.mean(dim=0, keepdim=True)
        return enc_words, enc_utt, att_toks

    def _encode_basic(self, utts_indexed):
        utt_data = batch_seqs(utts_indexed).to(self.device)
        enc_words, (enc_utt, _) = self.encoder(utt_data)
        enc_words = self.proj(enc_words)
        enc_utt = self.proj(torch.cat(enc_utt.split(1), dim=2))
        return enc_words, enc_utt, utts_indexed

    def train(self, real_utts, fake_utts, lfs):
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

        opt = optim.Adam(self.parameters(), lr=0.001)
        total_loss = 0
        for i in range(FLAGS.train_iters):
            if (i+1) % 10 == 0:
                print("{:.3f}".format(total_loss / 10), file=sys.stderr)
                sys.stderr.flush()
                total_loss = 0

            indices = np.random.randint(len(utts_indexed), size=FLAGS.batch_size)
            utts = [utts_indexed[i] for i in indices]
            lfs = [lfs_indexed[i] for i in indices]
            lf_data = batch_seqs(lfs).to(self.device)
            lf_ctx = lf_data[:-1, :]
            lf_tgt = lf_data[1:, :].view(-1)

            if FLAGS.seq2seq__pretrained_enc:
                enc_words, enc_utt, att_toks = self._encode_pretrained([utts_raw[i] for i in indices])
            else:
                enc_words, enc_utt, att_toks = self._encode_basic([utts_indexed[i] for i in indices])

            dec_state = (enc_utt, torch.zeros_like(enc_utt))
            logits, *_ = self.decoder(
                dec_state,
                lf_ctx.shape[0],
                ref_tokens=lf_ctx,
                att_features=(enc_words,),
                att_tokens=(att_toks,),
            )

            logits = logits.view(-1, logits.shape[-1])
            loss = self.loss(logits, lf_tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    def save(self, location):
        torch.save(self.state_dict(), location)

    def load(self, location):
        self.load_state_dict(torch.load(location))

    def predict(self, utt, gold_lf):
        utt_data = batch_seqs([self.vocab.encode(util.word_tokenize(utt), unk=True)]).to(self.device)
        enc_words, (enc_utt, _) = self.encoder(utt_data)
        enc_words = self.proj(enc_words)
        enc_utt = self.proj(torch.cat(enc_utt.split(1), dim=2))
        dec_state = (enc_utt, torch.zeros_like(enc_utt))
        preds = self.decoder.beam(
            dec_state,
            10,
            50,
            att_features=(enc_words,),
        )
        lfs = [util.lf_detokenize(self.vocab.decode(pred)) for pred in preds]
        lfs = [lf for lf in lfs if lf in self.lfs]
        if len(lfs) > 0:
            return lfs[0]
        return self.lfs[0]
