def lf_tokenize(vocab, lf):
    tokens = lf.replace("(", "( ").replace(")", " )").split()
    out = []
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
        out.append(vocab[token])
    return out

def lf_detokenize(reverse_vocab, tokens):
    tokens = [reverse_vocab[t] for t in tokens]
    lf = " ".join(tokens).replace("( ", "(").replace(" )", ")")
    return lf
